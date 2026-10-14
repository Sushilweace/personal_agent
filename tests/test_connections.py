"""Offline tests for the /connections page API (no Google, no network, no model).

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import connections
import gmail_access
from config import settings
from test_jobs import LINKEDIN_HTML, NAUKRI_HTML, PROFILE_TEXT, _message, fake_service

HEAD = {"X-Live-Client": "1"}
GOOD_CLIENT = json.dumps(
    {
        "installed": {
            "client_id": "id.apps.googleusercontent.com",
            "client_secret": "secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
)
GOOD_TOKEN = json.dumps(
    {"token": "t", "refresh_token": "r", "client_id": "c", "client_secret": "s", "token_uri": "https://oauth2.googleapis.com/token"}
)


class ConnectionsApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for name, value in (("SECRETS_DIR", root / "secrets"), ("JOB_PROFILE_FILE", root / "docs" / "job_profile.md")):
            patcher = mock.patch.object(settings, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        app = FastAPI()
        app.include_router(connections.create_router())
        self.client = TestClient(app, base_url=f"http://127.0.0.1:{settings.WEBUI_PORT}")

    # -- security ---------------------------------------------------------
    def test_guard(self):
        self.assertEqual(self.client.get("/api/connections").status_code, 403)  # no client header
        evil_host = self.client.get("/api/connections", headers={**HEAD, "Host": "evil.example.com"})
        self.assertEqual(evil_host.status_code, 403)
        evil_origin = self.client.post("/api/connections/gmail/disconnect", headers={**HEAD, "Origin": "https://evil.example.com"})
        self.assertEqual(evil_origin.status_code, 403)
        self.assertEqual(self.client.get("/api/connections", headers=HEAD).status_code, 200)

    def test_page(self):
        page = self.client.get("/connections")
        self.assertEqual(page.status_code, 200)
        self.assertNotIn("__CONFIG__", page.text)
        self.assertNotIn("__ASSISTANT_NAME__", page.text)

    # -- gmail ------------------------------------------------------------
    def test_status_flow(self):
        status = self.client.get("/api/connections", headers=HEAD).json()["gmail"]
        self.assertFalse(status["client_file"])
        self.assertFalse(status["connected"])

        bad = self.client.post("/api/connections/gmail/client", headers=HEAD, json={"content": "not json"})
        self.assertEqual(bad.status_code, 400)
        web = self.client.post("/api/connections/gmail/client", headers=HEAD, json={"content": json.dumps({"web": {}})})
        self.assertEqual(web.status_code, 400)
        self.assertIn("Desktop app", web.json()["detail"])
        incomplete = self.client.post("/api/connections/gmail/client", headers=HEAD, json={"content": json.dumps({"installed": {}})})
        self.assertEqual(incomplete.status_code, 400)
        self.assertFalse(gmail_access.credentials_file().exists())

        ok = self.client.post("/api/connections/gmail/client", headers=HEAD, json={"content": GOOD_CLIENT})
        self.assertEqual(ok.status_code, 200)
        self.assertTrue(self.client.get("/api/connections", headers=HEAD).json()["gmail"]["client_file"])

    def test_connect_requires_client_file(self):
        self.assertEqual(self.client.post("/api/connections/gmail/connect", headers=HEAD).status_code, 400)
        gmail_access.save_client_file(GOOD_CLIENT)
        with mock.patch.object(gmail_access, "start_sign_in", return_value=True) as start:
            res = self.client.post("/api/connections/gmail/connect", headers=HEAD)
        self.assertEqual(res.json(), {"started": True})
        start.assert_called_once()

    def test_connected_status_and_disconnect(self):
        gmail_access.save_client_file(GOOD_CLIENT)
        gmail_access.token_file().write_text(GOOD_TOKEN, encoding="utf-8")
        gmail_access.account_file().write_text("me@gmail.com", encoding="utf-8")
        status = self.client.get("/api/connections", headers=HEAD).json()["gmail"]
        self.assertTrue(status["connected"])
        self.assertEqual(status["email"], "me@gmail.com")

        with mock.patch("httpx.post", return_value=mock.Mock(status_code=200)) as post:
            res = self.client.post("/api/connections/gmail/disconnect", headers=HEAD)
        self.assertEqual(res.json(), {"ok": True, "revoked": True})
        self.assertEqual(post.call_args.kwargs["params"], {"token": "r"})  # refresh token revoked at Google
        self.assertFalse(gmail_access.token_file().exists())
        self.assertFalse(gmail_access.account_file().exists())
        self.assertFalse(self.client.get("/api/connections", headers=HEAD).json()["gmail"]["connected"])

    def test_disconnect_works_offline(self):
        gmail_access.token_file().parent.mkdir(parents=True, exist_ok=True)
        gmail_access.token_file().write_text(GOOD_TOKEN, encoding="utf-8")
        with mock.patch("httpx.post", side_effect=OSError("offline")):
            res = self.client.post("/api/connections/gmail/disconnect", headers=HEAD)
        self.assertEqual(res.json(), {"ok": True, "revoked": False})
        self.assertFalse(gmail_access.token_file().exists())

    def test_sign_in_background_states(self):
        gmail_access._signin.update(state="idle", message="")
        with mock.patch.object(gmail_access, "sign_in_blocking", side_effect=RuntimeError("closed window")):
            self.assertTrue(gmail_access.start_sign_in())
            for thread in __import__("threading").enumerate():
                if thread.name == "gmail-signin":
                    thread.join(5)
        state = gmail_access.sign_in_state()
        self.assertEqual(state["state"], "error")
        self.assertIn("not completed", state["message"])

    def test_test_endpoint_when_not_connected(self):
        res = self.client.post("/api/connections/gmail/test", headers=HEAD)
        self.assertEqual(res.status_code, 400)
        self.assertIn("not connected", res.json()["detail"].lower())

    def test_api_not_enabled_error_is_explained(self):
        from googleapiclient.errors import HttpError

        body = json.dumps(
            {"error": {"message": "Gmail API has not been used in project 418724036185 before or it is disabled.",
                       "errors": [{"reason": "accessNotConfigured"}]}}
        ).encode()
        error = HttpError(mock.Mock(status=403, reason="Forbidden"), body)
        text = gmail_access.describe_api_error(error)
        self.assertIn("not enabled", text)
        self.assertIn("project=418724036185", text)
        self.assertIsNone(gmail_access.describe_api_error(ValueError("x")))

        self.client.put("/api/connections/profile", headers=HEAD, json={"text": PROFILE_TEXT})
        with mock.patch.object(gmail_access, "build_service", side_effect=error):
            res = self.client.post("/api/connections/jobs", headers=HEAD, json={"days": 7})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Enable", res.json()["detail"])

    # -- profile ----------------------------------------------------------
    def test_profile_roundtrip(self):
        self.assertEqual(self.client.get("/api/connections/profile", headers=HEAD).json()["text"], "")
        saved = self.client.put("/api/connections/profile", headers=HEAD, json={"text": PROFILE_TEXT})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["summary"]["roles"], ["python developer", "backend engineer"])
        loaded = self.client.get("/api/connections/profile", headers=HEAD).json()
        self.assertEqual(loaded["text"], PROFILE_TEXT)
        too_big = self.client.put("/api/connections/profile", headers=HEAD, json={"text": "x" * 20_001})
        self.assertEqual(too_big.status_code, 422)

    # -- job check --------------------------------------------------------
    def test_jobs_endpoint(self):
        self.client.put("/api/connections/profile", headers=HEAD, json={"text": PROFILE_TEXT})
        store = {"1": _message("x", "a@linkedin.com", LINKEDIN_HTML), "2": _message("y", "b@naukri.com", NAUKRI_HTML)}
        service, _ = fake_service(store)
        with mock.patch.object(gmail_access, "build_service", return_value=service):
            res = self.client.post("/api/connections/jobs", headers=HEAD, json={"days": 7})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["emails"], 2)
        self.assertEqual(data["jobs"][0]["title"], "Python Developer")
        self.assertEqual(data["jobs"][0]["url"], "https://www.linkedin.com/comm/jobs/view/111")
        self.assertEqual(data["by_source"].get("LinkedIn"), 3)
        self.assertEqual(data["by_source"].get("Naukri"), 2)

    def test_jobs_endpoint_errors_are_readable(self):
        no_profile = self.client.post("/api/connections/jobs", headers=HEAD, json={"days": 7})
        self.assertEqual(no_profile.status_code, 400)
        self.assertIn("profile", no_profile.json()["detail"].lower())
        self.client.put("/api/connections/profile", headers=HEAD, json={"text": PROFILE_TEXT})
        not_connected = self.client.post("/api/connections/jobs", headers=HEAD, json={"days": 7})
        self.assertEqual(not_connected.status_code, 400)
        self.assertIn("Connect button", not_connected.json()["detail"])
        self.assertEqual(self.client.post("/api/connections/jobs", headers=HEAD, json={"days": 99}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
