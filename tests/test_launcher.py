"""Offline tests for the desktop-app launcher helpers and the Quit endpoint."""

from __future__ import annotations

import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import launcher
from config import settings


class PortTests(unittest.TestCase):
    def test_port_open(self):
        with socket.socket() as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen()
            port = srv.getsockname()[1]
            self.assertTrue(launcher.port_open("127.0.0.1", port))
        self.assertFalse(launcher.port_open("127.0.0.1", port, timeout=0.2))  # closed again


class WindowTests(unittest.TestCase):
    def test_app_window_uses_installed_browser(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp, "msedge.exe")
            exe.write_text("")
            with (
                mock.patch.object(launcher, "_browser_candidates", return_value=[Path(tmp, "nope.exe"), exe]),
                mock.patch.object(launcher.subprocess, "Popen") as popen,
                mock.patch.object(launcher.webbrowser, "open") as fallback,
                mock.patch.object(settings, "OPEN_AS_APP", True),
            ):
                launcher.open_window("http://127.0.0.1:7860/live")
        args = popen.call_args.args[0]
        self.assertEqual(args[0], str(exe))
        self.assertIn("--app=http://127.0.0.1:7860/live", args)
        fallback.assert_not_called()

    def test_falls_back_to_default_browser(self):
        for patches in ({"OPEN_AS_APP": False}, {"OPEN_AS_APP": True}):
            with (
                mock.patch.object(launcher, "find_app_browser", return_value=None),
                mock.patch.object(launcher.webbrowser, "open") as fallback,
                mock.patch.object(settings, "OPEN_AS_APP", patches["OPEN_AS_APP"]),
            ):
                launcher.open_window("http://127.0.0.1:7860/live")
            fallback.assert_called_once_with("http://127.0.0.1:7860/live")

    def test_open_when_ready_waits_for_server(self):
        with socket.socket() as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen()
            port = srv.getsockname()[1]
            with mock.patch.object(launcher, "open_window") as opener:
                launcher.open_when_ready(f"http://127.0.0.1:{port}/live", timeout=5).join(6)
        opener.assert_called_once_with(f"http://127.0.0.1:{port}/live")


class OllamaTests(unittest.TestCase):
    def test_already_running(self):
        with mock.patch.object(launcher, "port_open", return_value=True), mock.patch.object(launcher.subprocess, "Popen") as popen:
            self.assertTrue(launcher.ensure_ollama())
        popen.assert_not_called()

    def test_starts_local_ollama_then_waits(self):
        answers = iter([False, False, True])
        with (
            mock.patch.object(launcher, "port_open", side_effect=lambda *a, **k: next(answers)),
            mock.patch.object(launcher, "_find_ollama", return_value="ollama.exe"),
            mock.patch.object(launcher.subprocess, "Popen") as popen,
            mock.patch.object(launcher.time, "sleep"),
        ):
            self.assertTrue(launcher.ensure_ollama(wait_seconds=5))
        self.assertEqual(popen.call_args.args[0], ["ollama.exe", "serve"])

    def test_never_starts_a_remote_ollama(self):
        with (
            mock.patch.object(settings, "OLLAMA_BASE_URL", "http://192.168.1.50:11434"),
            mock.patch.object(launcher, "port_open", return_value=False),
            mock.patch.object(launcher.subprocess, "Popen") as popen,
        ):
            self.assertFalse(launcher.ensure_ollama())
        popen.assert_not_called()

    def test_autostart_can_be_disabled(self):
        with (
            mock.patch.object(settings, "OLLAMA_AUTOSTART", False),
            mock.patch.object(launcher, "port_open", return_value=False),
            mock.patch.object(launcher.subprocess, "Popen") as popen,
        ):
            self.assertFalse(launcher.ensure_ollama())
        popen.assert_not_called()


class QuitEndpointTests(unittest.TestCase):
    def setUp(self):
        import web

        # create_server needs a graph + Gradio; test only the quit route on a bare app.
        self.app = FastAPI()
        with mock.patch.object(web.gr, "mount_gradio_app", side_effect=lambda app, *a, **k: app), mock.patch.object(
            web, "create_app", return_value=None
        ), mock.patch.object(web.live, "create_router", return_value=__import__("fastapi").APIRouter()):
            self.app = web.create_server(mock.Mock(), mock.Mock())
        self.client = TestClient(self.app, base_url=f"http://127.0.0.1:{settings.WEBUI_PORT}")

    def test_quit_stops_server_only_for_our_own_page(self):
        server = mock.Mock(should_exit=False)
        self.app.state.server = server
        self.assertEqual(self.client.post("/api/quit").status_code, 403)  # no client header
        self.assertEqual(self.client.post("/api/quit", headers={"X-Live-Client": "1", "Origin": "https://evil.example.com"}).status_code, 403)
        self.assertFalse(server.should_exit)
        ok = self.client.post("/api/quit", headers={"X-Live-Client": "1"})
        self.assertEqual(ok.json(), {"ok": True})
        self.assertTrue(server.should_exit)


if __name__ == "__main__":
    unittest.main()
