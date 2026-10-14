"""Tool registry for the personal agent."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from tools.jobs_tools import read_job_alerts
from tools.notes_tools import query_personal_notes
from tools.system_tools import get_current_time, get_system_info, run_calculator
from tools.web_tools import fetch_webpage, open_in_browser, web_search

ALL_TOOLS: list[BaseTool] = [
    query_personal_notes,
    run_calculator,
    get_current_time,
    get_system_info,
    web_search,
    fetch_webpage,
    open_in_browser,
    read_job_alerts,
]

__all__ = [
    "ALL_TOOLS",
    "fetch_webpage",
    "get_current_time",
    "get_system_info",
    "open_in_browser",
    "query_personal_notes",
    "read_job_alerts",
    "run_calculator",
    "web_search",
]
