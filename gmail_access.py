"""Read-only Gmail access for job-alert emails.

Safety design:
* Only the ``gmail.readonly`` scope is ever requested: this code cannot send, delete or change
  mail, and neither can anyone who steals the token.
* No passwords: you sign in on Google's own page once (``python gmail_auth.py``); only a
  revocable token is stored locally in ``secrets/`` (git-ignored).
* The model never writes Gmail queries. The query is fixed here: recent mail from the job-site
  senders listed in ``JOB_ALERT_SENDERS``. The rest of your inbox is never read.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from typing import Any

from config import settings
from jobs import AlertEmail

logger = logging.getLogger(__name__)

SCOPES: list[str] = ["https://www.googleapis.com/auth/gmail.readonly"]
_MAX_BODY_CHARS = 600_000
_HTTP_TIMEOUT = 20

SETUP_HINT = "Gmail is not connected yet. Follow the 'Job alerts from Gmail' steps in README.md, then run: python gmail_auth.py"


class GmailNotConnected(RuntimeError):
    """No usable Google sign-in is stored (never set up, revoked or expired)."""


def credentials_file():
    return settings.SECRETS_DIR / "google_credentials.json"


def token_file():
    return settings.SECRETS_DIR / "gmail_token.json"


def account_file():
    return settings.SECRETS_DIR / "gmail_account.txt"


def sender_domains() -> list[str]:
    return [d.strip().lower() for d in settings.JOB_ALERT_SENDERS.split(",") if d.strip()]


def build_query(days: int) -> str:
    senders = " OR ".join(sender_domains())
    return f"newer_than:{days}d from:({senders})"


def describe_api_error(exc: Exception) -> str | None:
    """A plain-words explanation for common Gmail API setup errors, or None if not recognised."""
    try:
        from googleapiclient.errors import HttpError
    except ImportError:
        return None
    if not isinstance(exc, HttpError):
        return None
    text = str(exc)
    status = getattr(exc.resp, "status", 0)
    if "accessNotConfigured" in text or "has not been used in project" in text:
        import re

        match = re.search(r"project[= ](\d{6,})", text)
        link = "https://console.developers.google.com/apis/api/gmail.googleapis.com/overview"
        if match:
            link += f"?project={match.group(1)}"
        return f"The Gmail API is not enabled in your Google Cloud project. Open {link} , press Enable, wait a minute, then try again."
    if status == 401 or "invalid_grant" in text:
        return "Google rejected the saved sign-in. Disconnect and press Connect Gmail again."
    if status == 429:
        return "Google says too many requests right now. Wait a minute and try again."
    return None


def get_credentials():
    """Stored credentials, refreshed if needed. Raises GmailNotConnected if sign-in is required."""
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    path = token_file()
    if not path.exists():
        raise GmailNotConnected(SETUP_HINT)
    creds = Credentials.from_authorized_user_file(str(path), SCOPES)
    if not creds.valid:
        if creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as exc:
                raise GmailNotConnected(
                    "Gmail sign-in expired or was revoked. Run: python gmail_auth.py"
                ) from exc
            path.write_text(creds.to_json(), encoding="utf-8")
        else:
            raise GmailNotConnected("Gmail sign-in is incomplete. Run: python gmail_auth.py")
    return creds


def build_service(creds=None):
    import google_auth_httplib2
    import httplib2
    from googleapiclient.discovery import build

    creds = creds or get_credentials()
    http = google_auth_httplib2.AuthorizedHttp(creds, http=httplib2.Http(timeout=_HTTP_TIMEOUT))
    return build("gmail", "v1", http=http, cache_discovery=False)


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")


def _collect_bodies(part: dict[str, Any], out: dict[str, list[str]]) -> None:
    """Walk the MIME tree and gather text/html and text/plain bodies."""
    mime = part.get("mimeType", "")
    data = (part.get("body") or {}).get("data")
    if data and mime in ("text/html", "text/plain") and len(data) < _MAX_BODY_CHARS:
        out["html" if mime == "text/html" else "text"].append(_decode(data))
    for sub in part.get("parts") or []:
        _collect_bodies(sub, out)


def message_to_email(message: dict[str, Any]) -> AlertEmail:
    payload = message.get("payload") or {}
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    bodies: dict[str, list[str]] = {"html": [], "text": []}
    _collect_bodies(payload, bodies)
    return AlertEmail(
        subject=headers.get("subject", ""),
        sender=headers.get("from", ""),
        date=headers.get("date", ""),
        html="\n".join(bodies["html"]),
        text="\n".join(bodies["text"]),
    )


def fetch_job_alert_emails(days: int, max_emails: int, service=None) -> list[AlertEmail]:
    """Recent job-alert emails (newest first). ``service`` can be injected for tests."""
    service = service or build_service()
    users = service.users().messages()
    listing = users.list(userId="me", q=build_query(days), maxResults=max_emails).execute()
    emails: list[AlertEmail] = []
    for ref in listing.get("messages", []):
        message = users.get(userId="me", id=ref["id"], format="full").execute()
        emails.append(message_to_email(message))
    logger.info("Fetched %d job-alert emails (last %d days)", len(emails), days)
    return emails


# --------------------------------------------------------------------------- #
# Connecting / disconnecting (used by gmail_auth.py and the /connections page)
# --------------------------------------------------------------------------- #
MAX_CLIENT_FILE_BYTES = 20_000


def save_client_file(content: str) -> None:
    """Validate and store the OAuth client JSON downloaded from Google Cloud ("Desktop app")."""
    if len(content.encode("utf-8")) > MAX_CLIENT_FILE_BYTES:
        raise ValueError("That file is too large to be a Google client file.")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("That is not a valid JSON file.") from exc
    if not isinstance(data, dict):
        raise ValueError("That is not a Google client file.")
    if "installed" not in data:
        kind = next(iter(data), "unknown")
        raise ValueError(
            f"This is a '{kind}' client. Create the OAuth client with application type 'Desktop app' and download that JSON."
        )
    installed = data["installed"]
    if not isinstance(installed, dict) or not all(
        isinstance(installed.get(k), str) and installed[k] for k in ("client_id", "client_secret", "auth_uri", "token_uri")
    ):
        raise ValueError("The client file is missing client_id, client_secret, auth_uri or token_uri.")
    settings.SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    credentials_file().write_text(content, encoding="utf-8")


def account_email(creds=None, service=None) -> str:
    service = service or build_service(creds)
    return str(service.users().getProfile(userId="me").execute().get("emailAddress", ""))


def sign_in_blocking(timeout_seconds: int | None = None) -> str:
    """Open Google's sign-in page and wait for consent. Returns the Gmail address."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not credentials_file().exists():
        raise GmailNotConnected("The Google client file is missing. Add it first (see README.md, 'Job alerts from Gmail').")
    settings.SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file()), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", open_browser=True, timeout_seconds=timeout_seconds)
    token_file().write_text(creds.to_json(), encoding="utf-8")
    email = account_email(creds)
    account_file().write_text(email, encoding="utf-8")
    return email


_signin_lock = threading.Lock()
_signin: dict[str, str] = {"state": "idle", "message": ""}  # idle | waiting | done | error


def sign_in_state() -> dict[str, str]:
    return dict(_signin)


def start_sign_in(timeout_seconds: int = 180) -> bool:
    """Start the Google sign-in in the background. False if one is already waiting."""
    with _signin_lock:
        if _signin["state"] == "waiting":
            return False
        _signin.update(state="waiting", message="")

    def work() -> None:
        try:
            email = sign_in_blocking(timeout_seconds)
            _signin.update(state="done", message=email)
        except GmailNotConnected as exc:
            _signin.update(state="error", message=str(exc))
        except Exception as exc:  # consent denied, window closed, timeout, API not enabled...
            logger.warning("Gmail sign-in did not complete: %s: %s", type(exc).__name__, exc)
            _signin.update(state="error", message=f"Sign-in was not completed ({type(exc).__name__}). Try again.")

    threading.Thread(target=work, daemon=True, name="gmail-signin").start()
    return True


def status() -> dict[str, Any]:
    """Local-only status (no network): is the client file there, is a usable token stored?"""
    connected = False
    if token_file().exists():
        try:
            from google.oauth2.credentials import Credentials

            creds = Credentials.from_authorized_user_file(str(token_file()), SCOPES)
            connected = bool(creds.refresh_token or creds.valid)
        except Exception:
            connected = False
    email = ""
    if connected and account_file().exists():
        email = account_file().read_text(encoding="utf-8").strip()
    return {"client_file": credentials_file().exists(), "connected": connected, "email": email}


def disconnect() -> bool:
    """Delete the local token and ask Google to revoke it. Returns True if Google confirmed."""
    revoked = False
    if token_file().exists():
        try:
            import httpx

            data = json.loads(token_file().read_text(encoding="utf-8"))
            token = data.get("refresh_token") or data.get("token")
            if token:
                revoked = httpx.post("https://oauth2.googleapis.com/revoke", params={"token": token}, timeout=10).status_code == 200
        except Exception as exc:
            logger.warning("Could not revoke the Google token: %s", exc)
        token_file().unlink(missing_ok=True)
    account_file().unlink(missing_ok=True)
    return revoked
