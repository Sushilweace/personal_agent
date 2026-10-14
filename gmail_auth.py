"""One-time Google sign-in for the job-alert feature (read-only Gmail).

    python gmail_auth.py            # sign in (opens your browser on Google's own page)
    python gmail_auth.py --check    # test the stored sign-in
    python gmail_auth.py --logout   # delete the local token (also revoke it at
                                    # https://myaccount.google.com/permissions)
"""

from __future__ import annotations

import argparse
import sys

from config import settings
from gmail_access import (
    GmailNotConnected,
    build_service,
    disconnect,
    fetch_job_alert_emails,
    sign_in_blocking,
    token_file,
)


def sign_in() -> int:
    print("Your browser will open Google's sign-in page. Jadu asks for READ-ONLY access to your Gmail.")
    try:
        email = sign_in_blocking()
    except GmailNotConnected as exc:
        print(exc)
        return 1
    print(f"Signed in as {email}. Token saved to {token_file()}")
    return check()


def check() -> int:
    try:
        service = build_service()
        profile = service.users().getProfile(userId="me").execute()
        emails = fetch_job_alert_emails(settings.JOB_ALERT_DAYS, 5, service=service)
    except GmailNotConnected as exc:
        print(exc)
        return 1
    except Exception as exc:  # network, API not enabled, ...
        print(f"Check failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"Connected as {profile.get('emailAddress')} (read-only). Found {len(emails)} recent job-alert emails (max 5 counted).")
    return 0


def logout() -> int:
    revoked = disconnect()
    print("Local token deleted." + (" Access revoked at Google." if revoked else " Also revoke access at https://myaccount.google.com/permissions"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="test the stored sign-in")
    group.add_argument("--logout", action="store_true", help="delete the local token")
    args = parser.parse_args()
    if args.check:
        return check()
    if args.logout:
        return logout()
    return sign_in()


if __name__ == "__main__":
    sys.exit(main())
