"""Web tools: DuckDuckGo search, a guarded page fetcher and a browser launcher."""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import warnings
import webbrowser
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from config import settings

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES: int = 1_000_000
MAX_REDIRECTS: int = 3
MAX_SNIPPET_CHARS: int = 300
USER_AGENT: str = "Mozilla/5.0 (compatible; PersonalAgent/1.0; +local)"
_TEXT_TAGS_TO_DROP: tuple[str, ...] = (
    "script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe",
)


def _load_ddgs() -> Any:
    """Import the DuckDuckGo client (new ``ddgs`` name first, then the legacy package)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # duckduckgo_search warns about its rename
        try:
            from ddgs import DDGS  # type: ignore[import-not-found]
        except ImportError:
            from duckduckgo_search import DDGS
    return DDGS


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " … [truncated]"


def _assert_public_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts that resolve to private/local addresses.

    The model chooses the URL, so this prevents it (or a prompt-injected page)
    from probing localhost or the user's LAN.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http:// and https:// URLs are supported")
    host = parsed.hostname
    if not host:
        raise ValueError("URL has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve host '{host}'") from exc
    for info in addresses:
        ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        if not ip.is_global:
            raise ValueError("access to private or local network addresses is blocked")


def _html_to_text(html: str) -> tuple[str, str]:
    """Return ``(title, readable_text)`` extracted from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_TEXT_TAGS_TO_DROP):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return title, text


@tool
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo for current or external information (news, facts, documentation).
    Returns the top results with title, URL and a short snippet."""
    query = (query or "").strip()
    if not query:
        return "Error: the search query is empty."
    try:
        ddgs_cls = _load_ddgs()
        results = list(ddgs_cls().text(query, max_results=settings.WEB_MAX_RESULTS))
    except Exception as exc:
        logger.warning("web_search failed: %s", exc)
        return f"Error: web search failed ({type(exc).__name__}: {exc}). Try again later or answer without it."

    if not results:
        return f"No web results found for '{query}'."

    lines: list[str] = []
    for rank, item in enumerate(results, start=1):
        title = str(item.get("title", "")).strip() or "(no title)"
        url = str(item.get("href") or item.get("url") or "").strip()
        snippet = _truncate(str(item.get("body", "")).strip(), MAX_SNIPPET_CHARS)
        lines.append(f"{rank}. {title}\n   {url}\n   {snippet}")
    return "\n".join(lines)


_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}(?:[:/?#].*)?$", re.IGNORECASE)
SEARCH_URL: str = "https://duckduckgo.com/?q={}"
MAX_TARGET_CHARS: int = 500


@tool
def open_in_browser(target: str) -> str:
    """Open a page in the user's own default web browser (visible on their screen).
    `target` is a full URL, a domain such as 'youtube.com', or plain search words (opens a web search).
    Use when the user asks to open, launch or go to a website, or to search in the browser.
    This does NOT return the page content; use web_search or fetch_webpage to read information yourself."""
    target = (target or "").strip()
    if not target:
        return "Error: nothing to open."
    if len(target) > MAX_TARGET_CHARS:
        return f"Error: the target is too long (max {MAX_TARGET_CHARS} characters)."
    try:
        if _SCHEME_RE.match(target):
            url = target
        elif _DOMAIN_RE.match(target):
            url = f"https://{target}"
        else:
            url = SEARCH_URL.format(quote_plus(target))
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return "Error: only http:// and https:// addresses can be opened."
        if not webbrowser.open(url, new=2):
            return "Error: no web browser could be launched on this computer."
        return f"Opened {url} in the default browser."
    except Exception as exc:
        logger.exception("open_in_browser failed")
        return f"Error: could not open the browser ({type(exc).__name__}: {exc})"


@tool
def fetch_webpage(url: str) -> str:
    """Download a public web page (http/https URL) and return its readable text.
    Use after web_search to read a specific result in more detail."""
    url = (url or "").strip()
    if not url:
        return "Error: the URL is empty."
    try:
        current = url
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"}
        with httpx.Client(
            timeout=settings.WEB_TIMEOUT_SECONDS, follow_redirects=False, headers=headers
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                _assert_public_url(current)  # re-validated on every redirect hop
                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("redirect without a Location header")
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not (
                        content_type.startswith("text/") or "html" in content_type or "xml" in content_type
                    ):
                        raise ValueError(f"unsupported content type: {content_type.split(';')[0]}")
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) >= MAX_DOWNLOAD_BYTES:
                            break
                    raw = bytes(body[:MAX_DOWNLOAD_BYTES]).decode(response.encoding or "utf-8", errors="replace")
                    break
            else:
                return "Error: too many redirects."

        if "html" in content_type or raw.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            title, text = _html_to_text(raw)
        else:
            title, text = "", raw.strip()
        if not text:
            return f"The page at {current} contained no readable text."
        header = f"URL: {current}\n" + (f"Title: {title}\n" if title else "")
        return header + "\n" + _truncate(text, settings.WEB_MAX_CHARS)
    except httpx.HTTPStatusError as exc:
        return f"Error: the server returned HTTP {exc.response.status_code} for {url}."
    except httpx.TimeoutException:
        return f"Error: timed out fetching {url}."
    except (httpx.HTTPError, ValueError) as exc:
        return f"Error: could not fetch {url} ({type(exc).__name__}: {exc})"
    except Exception as exc:
        logger.exception("fetch_webpage failed")
        return f"Error: unexpected failure fetching {url} ({type(exc).__name__}: {exc})"
