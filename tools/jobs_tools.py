"""Tool: rank the user's Gmail job-alert emails (LinkedIn, Naukri, ...) against their requirements."""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from config import settings
from gmail_access import GmailNotConnected, describe_api_error, fetch_job_alert_emails, sender_domains
from jobs import Profile, Report, format_report, load_profile, rank_jobs, report_links

logger = logging.getLogger(__name__)


class JobCheckError(Exception):
    """A problem to report to the user in plain words (not a bug)."""

    def __init__(self, message: str, not_connected: bool = False) -> None:
        super().__init__(message)
        self.not_connected = not_connected


def run_job_check(days: int) -> tuple[Report, Profile, int]:
    """Fetch the job-alert emails and rank them. Shared by the chat tool and the /connections page.

    Returns ``(report, profile, days)``; ``report.emails == 0`` means no alert emails were found.
    Raises JobCheckError with a readable message otherwise."""
    profile = load_profile(settings.JOB_PROFILE_FILE)
    if profile is None:
        raise JobCheckError(f"The job profile was not found at {settings.JOB_PROFILE_FILE}. Create it first.")
    if profile.is_empty:
        raise JobCheckError(
            f"The job profile ({settings.JOB_PROFILE_FILE.name}) has no target roles or must-have skills yet. Fill it in first."
        )
    days = max(1, min(int(days), 30))
    try:
        emails = fetch_job_alert_emails(days, settings.JOB_MAX_EMAILS)
    except GmailNotConnected as exc:
        raise JobCheckError(str(exc), not_connected=True) from exc
    except Exception as exc:  # network, quota, API disabled...
        logger.exception("Job check failed")
        friendly = describe_api_error(exc)
        if friendly:
            raise JobCheckError(friendly) from exc
        raise JobCheckError(f"Could not read Gmail ({type(exc).__name__}: {exc})") from exc
    return rank_jobs(emails, profile, sender_domains(), settings.JOB_TOP_N), profile, days


@tool(response_format="content_and_artifact")
def read_job_alerts(days: int = 7) -> tuple[str, list[dict[str, str]]]:
    """Find the best job matches in the user's Gmail job-alert emails (LinkedIn, Naukri, Indeed and
    similar), ranked against the requirements in the user's job profile file.
    Use only when the user asks about job alerts, job emails or jobs matching their requirements.
    'days' is how far back to look (default 7). Read-only: it never sends, deletes or applies.
    Links are attached to the answer automatically, so do not write any URLs yourself."""
    try:
        report, profile, days = run_job_check(days)
    except JobCheckError as exc:
        return f"Error: {exc}", []
    if report.emails == 0:
        return f"No job-alert emails from the last {days} days (looked at: {', '.join(sender_domains())}).", []
    return format_report(report, profile, days, urls=False), report_links(report)
