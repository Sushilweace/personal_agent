"""Helpers that make Jadu feel like a desktop app.

* start Ollama in the background if it is not running,
* open Jadu in its own window (Edge/Chrome ``--app`` mode: no tabs, no address bar),
* detect that Jadu is already running, so a second double-click just opens the window.

Nothing here opens a network port or changes what the server exposes.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

from config import settings

logger = logging.getLogger(__name__)

WINDOW_SIZE = "560,860"


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """True if something accepts connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# App-style window
# --------------------------------------------------------------------------- #
def _browser_candidates() -> list[Path]:
    paths: list[Path] = []
    if sys.platform == "win32":
        for var in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
            base = os.environ.get(var)
            if base:
                paths.append(Path(base, "Microsoft", "Edge", "Application", "msedge.exe"))
                paths.append(Path(base, "Google", "Chrome", "Application", "chrome.exe"))
    elif sys.platform == "darwin":
        paths += [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        ]
    for name in ("msedge", "microsoft-edge", "google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            paths.append(Path(found))
    return paths


def find_app_browser() -> Path | None:
    """The first installed Chromium-based browser that supports ``--app`` windows."""
    return next((p for p in _browser_candidates() if p.is_file()), None)


def open_window(url: str) -> None:
    """Open ``url`` as an app window if possible, otherwise in the default browser."""
    if settings.OPEN_AS_APP:
        browser = find_app_browser()
        if browser is not None:
            try:
                subprocess.Popen(
                    [str(browser), f"--app={url}", f"--window-size={WINDOW_SIZE}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except OSError as exc:
                logger.warning("Could not start %s: %s", browser, exc)
    webbrowser.open(url)


def open_when_ready(url: str, timeout: float = 90.0) -> threading.Thread:
    """Open the window as soon as the server accepts connections (models may take a while to load)."""
    parsed = urlparse(url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 80

    def work() -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if port_open(host, port):
                open_window(url)
                return
            time.sleep(0.3)

    thread = threading.Thread(target=work, daemon=True, name="open-window")
    thread.start()
    return thread


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
def _ollama_host_port() -> tuple[str, int]:
    parsed = urlparse(settings.OLLAMA_BASE_URL)
    return parsed.hostname or "localhost", parsed.port or 11434


def _find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidate = Path(local, "Programs", "Ollama", "ollama.exe")
        if candidate.is_file():
            return str(candidate)
    return None


def ensure_ollama(wait_seconds: float = 25.0) -> bool:
    """Make sure Ollama answers. Starts ``ollama serve`` (hidden) only for a local Ollama."""
    host, port = _ollama_host_port()
    if port_open(host, port):
        return True
    if host not in ("localhost", "127.0.0.1", "::1") or not settings.OLLAMA_AUTOSTART:
        return False
    exe = _find_ollama()
    if exe is None:
        return False
    print("Starting Ollama...")
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS  # keeps running after Jadu closes
    try:
        subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
    except OSError as exc:
        logger.warning("Could not start Ollama: %s", exc)
        return False
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if port_open(host, port):
            return True
        time.sleep(0.4)
    return False
