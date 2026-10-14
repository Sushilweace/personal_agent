"""Tool exposing semantic search over the user's personal documents."""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from config import settings
from rag import query_knowledge_base

logger = logging.getLogger(__name__)


@tool
def query_personal_notes(query: str) -> str:
    """Search the user's personal notes and documents (markdown, text, PDF) by meaning.
    Use for questions about the user's own files, projects, plans, tasks or saved information.
    The query should be a short natural-language description of what to find."""
    try:
        return query_knowledge_base(query, settings.RETRIEVAL_TOP_K)
    except Exception as exc:  # query_knowledge_base should not raise; belt and braces
        logger.exception("query_personal_notes failed")
        return f"Error: could not search personal notes ({type(exc).__name__}: {exc})"
