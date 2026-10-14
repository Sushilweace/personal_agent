"""The /connections page: connect Gmail (read-only), see LinkedIn/Naukri status, edit job
requirements and run a job check, all from the browser.

LinkedIn and Naukri have no official API for personal accounts, and logging in for you would
mean storing your passwords and breaking their terms (risking your account). So they are
connected through the job-alert emails they send to your Gmail instead.

All endpoints use the same localhost/Origin/header guard as the live voice API.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import gmail_access
from config import settings
from jobs import label_for, parse_profile
from live import _guard
from tools.jobs_tools import JobCheckError, run_job_check

logger = logging.getLogger(__name__)

NOT_CONNECTED_MSG = "Gmail is not connected. Use the Connect button in the Gmail card above."
PAGE_PATH = Path(__file__).resolve().parent / "static" / "connections.html"
MAX_PROFILE_CHARS = 20_000


class ClientFileBody(BaseModel):
    content: str = Field(min_length=2, max_length=gmail_access.MAX_CLIENT_FILE_BYTES)


class ProfileBody(BaseModel):
    text: str = Field(max_length=MAX_PROFILE_CHARS)


class JobsBody(BaseModel):
    days: int = Field(default=7, ge=1, le=30)


def _profile_summary(text: str) -> dict[str, list[str]]:
    p = parse_profile(text)
    return {"roles": p.roles, "must": p.must, "nice": p.nice, "locations": p.locations, "avoid": p.avoid}


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/connections", response_class=HTMLResponse)
    def page() -> HTMLResponse:
        config = {"name": settings.ASSISTANT_NAME, "senders": gmail_access.sender_domains()}
        text = PAGE_PATH.read_text(encoding="utf-8")
        text = text.replace("__ASSISTANT_NAME__", html.escape(settings.ASSISTANT_NAME))
        text = text.replace("__CONFIG__", json.dumps(config).replace("</", "<\\/"))
        return HTMLResponse(text, headers={"Cache-Control": "no-store"})

    @router.get("/api/connections")
    async def get_status(request: Request) -> JSONResponse:
        _guard(request)
        gmail = await run_in_threadpool(gmail_access.status)
        gmail["signin"] = gmail_access.sign_in_state()
        return JSONResponse({"gmail": gmail})

    @router.post("/api/connections/gmail/client")
    async def upload_client_file(request: Request, body: ClientFileBody) -> JSONResponse:
        _guard(request)
        try:
            gmail_access.save_client_file(body.content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"ok": True})

    @router.post("/api/connections/gmail/connect")
    async def connect(request: Request) -> JSONResponse:
        """Opens Google's sign-in page in your browser; poll /api/connections for the result."""
        _guard(request)
        if not gmail_access.credentials_file().exists():
            raise HTTPException(status_code=400, detail="Add the Google client file first.")
        started = gmail_access.start_sign_in()
        return JSONResponse({"started": started})

    @router.post("/api/connections/gmail/disconnect")
    async def disconnect(request: Request) -> JSONResponse:
        _guard(request)
        revoked = await run_in_threadpool(gmail_access.disconnect)
        return JSONResponse({"ok": True, "revoked": revoked})

    @router.post("/api/connections/gmail/test")
    async def test_connection(request: Request) -> JSONResponse:
        _guard(request)

        def work() -> dict:
            service = gmail_access.build_service()
            email = gmail_access.account_email(service=service)
            emails = gmail_access.fetch_job_alert_emails(settings.JOB_ALERT_DAYS, 10, service=service)
            gmail_access.account_file().write_text(email, encoding="utf-8")
            return {"email": email, "alerts": len(emails), "days": settings.JOB_ALERT_DAYS}

        try:
            return JSONResponse(await run_in_threadpool(work))
        except gmail_access.GmailNotConnected as exc:
            raise HTTPException(status_code=400, detail=NOT_CONNECTED_MSG) from exc
        except Exception as exc:
            logger.warning("Gmail test failed: %s", exc)
            detail = gmail_access.describe_api_error(exc) or f"Gmail check failed ({type(exc).__name__}). Is the Gmail API enabled?"
            raise HTTPException(status_code=502, detail=detail) from exc

    @router.get("/api/connections/profile")
    async def get_profile(request: Request) -> JSONResponse:
        _guard(request)
        path = settings.JOB_PROFILE_FILE
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        return JSONResponse({"text": text, "summary": _profile_summary(text), "file": path.name})

    @router.put("/api/connections/profile")
    async def save_profile(request: Request, body: ProfileBody) -> JSONResponse:
        _guard(request)
        path = settings.JOB_PROFILE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(body.text, encoding="utf-8")
        tmp.replace(path)
        return JSONResponse({"ok": True, "summary": _profile_summary(body.text)})

    @router.post("/api/connections/jobs")
    async def check_jobs(request: Request, body: JobsBody) -> JSONResponse:
        """Run the job check right now (no chat model involved) and return the ranked jobs."""
        _guard(request)
        try:
            report, _profile, days = await run_in_threadpool(run_job_check, body.days)
        except JobCheckError as exc:
            raise HTTPException(status_code=400, detail=NOT_CONNECTED_MSG if exc.not_connected else str(exc)) from exc
        return JSONResponse(
            {
                "days": days,
                "emails": report.emails,
                "total": report.total_jobs,
                "skipped": report.skipped,
                "weak": report.weak,
                "by_source": report.by_source,
                "jobs": [
                    {
                        "title": j.title,
                        "snippet": j.snippet,
                        "source": j.source,
                        "score": j.score,
                        "label": label_for(j.score),
                        "reasons": j.reasons,
                        "url": j.url,
                    }
                    for j in report.top
                ],
            }
        )

    return router
