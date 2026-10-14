"""Live voice conversation backend (served by ``web.py`` at ``/live``).

Flow, as in a "Gemini Live"-style chat:

    browser mic --(auto-detects speech start/stop)--> POST /api/stt  (local Whisper)
    transcript  --> POST /api/live-chat  (agent reply streamed sentence by sentence)
    sentences   --> browser text-to-speech, spoken as soon as the first one arrives

Security: the server only listens on localhost, and every request must carry the
custom ``X-Live-Client`` header and a localhost ``Host``/``Origin``. That blocks other
websites open in your browser (cross-site or DNS-rebinding requests) from driving the
agent, which can open pages on this computer.
"""

from __future__ import annotations

import html
import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import voice
from config import settings
from graph import _DEGENERATE_REPLIES

logger = logging.getLogger("personal_agent.live")

PAGE_PATH: Path = Path(__file__).resolve().parent / "static" / "live.html"
MAX_PCM_BYTES: int = 16_000 * 2 * 30  # 30 s of 16 kHz 16-bit mono
MIN_PCM_BYTES: int = 16_000 * 2 // 4  # 0.25 s
MIN_SENTENCE_CHARS: int = 6
CLIENT_HEADER: str = "x-live-client"


# --------------------------------------------------------------------------- #
# Sentence splitting + cleanup for speech
# --------------------------------------------------------------------------- #
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+|\n+")
_EMOJI = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf\ufe0f]")


class SentenceSplitter:
    """Collects streamed tokens and releases complete sentences as soon as they end."""

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        sentences: list[str] = []
        start = 0
        for match in _SENTENCE_END.finditer(self.buffer):
            candidate = self.buffer[start : match.start()].strip()
            if len(candidate) >= MIN_SENTENCE_CHARS:
                sentences.append(candidate)
                start = match.end()
        self.buffer = self.buffer[start:]
        return sentences

    def flush(self) -> list[str]:
        """Release whatever is left (dropping an empty-JSON-style non-answer)."""
        rest, self.buffer = self.buffer.strip(), ""
        return [rest] if rest and rest.lower() not in _DEGENERATE_REPLIES else []


def speakable(text: str) -> str:
    """Strip markdown, links and emoji so text-to-speech reads naturally."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "link", text)
    text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`#>~|]+", "", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else str(block.get("text", ""))
        for block in content
        if isinstance(block, (str, dict))
    )


def _event(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- #
# Agent reply stream
# --------------------------------------------------------------------------- #
def stream_reply(graph: CompiledStateGraph, thread_id: str, text: str) -> Iterator[bytes]:
    """Run the agent and yield NDJSON events: ``tool``, ``tool_result``, ``sentence``, ``error``, ``done``."""
    config = {
        "configurable": {"thread_id": thread_id, "voice": True},
        "recursion_limit": settings.MAX_GRAPH_STEPS,
    }
    splitter = SentenceSplitter()
    spoke = False  # has any speakable text been sent since the last tool call?
    current_id: str | None = None
    started = time.perf_counter()

    def emit(parts: list[str]) -> Iterator[bytes]:
        nonlocal spoke
        for part in parts:
            speak = speakable(part)
            if speak:
                spoke = True
                yield _event({"type": "sentence", "text": part, "speak": speak})

    try:
        for mode, payload in graph.stream(
            {"messages": [HumanMessage(content=text)]}, config, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessageChunk):
                    continue
                if chunk.id != current_id:  # a new model call started (e.g. the no-tools retry)
                    yield from emit(splitter.flush())
                    current_id = chunk.id
                piece = _content_text(chunk.content)
                if piece:
                    yield from emit(splitter.feed(piece))
            elif mode == "updates":
                for node, update in payload.items():
                    if not isinstance(update, dict):
                        continue
                    for message in update.get("messages", []):
                        if node == "agent" and isinstance(message, AIMessage):
                            yield from emit(splitter.flush())
                            current_id = None
                            if message.tool_calls:
                                spoke = False
                                for call in message.tool_calls:
                                    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                                    yield _event({"type": "tool", "name": call["name"], "args": _short(args, 100)})
                            elif not spoke:
                                # Nothing was streamed (non-streaming model, or an app-generated notice).
                                full = _content_text(message.content).strip()
                                if full.lower() not in _DEGENERATE_REPLIES:
                                    again = SentenceSplitter()
                                    yield from emit([*again.feed(full), *again.flush()])
                        elif isinstance(message, ToolMessage):
                            yield _event(
                                {"type": "tool_result", "name": message.name or "", "text": _short(_content_text(message.content), 300)}
                            )
        yield from emit(splitter.flush())
    except GraphRecursionError:
        yield _event({"type": "error", "text": "I hit my step limit without finishing. Please try a simpler request."})
    except Exception as exc:
        logger.exception("Live chat failed")
        yield _event({"type": "error", "text": f"Sorry, something went wrong ({type(exc).__name__})."})
    yield _event({"type": "done", "ms": round((time.perf_counter() - started) * 1000)})


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    thread_id: str = Field(min_length=4, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


def _allowed_hosts() -> set[str]:
    port = settings.WEBUI_PORT
    return {f"127.0.0.1:{port}", f"localhost:{port}", f"{settings.WEBUI_HOST}:{port}"}


def _guard(request: Request) -> None:
    """Reject requests that did not come from our own page on localhost."""
    allowed = _allowed_hosts()
    if (request.headers.get("host") or "").lower() not in allowed:
        raise HTTPException(status_code=403, detail="forbidden host")
    origin = request.headers.get("origin")
    if origin and origin.lower() not in {f"http://{host}" for host in allowed}:
        raise HTTPException(status_code=403, detail="forbidden origin")
    if request.headers.get(CLIENT_HEADER) != "1":
        raise HTTPException(status_code=403, detail="missing client header")


# --------------------------------------------------------------------------- #
# Indian neural voices (edge-tts: Microsoft's online "Natural" voices)
# --------------------------------------------------------------------------- #
MAX_TTS_CHARS: int = 600
VOICE_LIST_TTL_SECONDS: float = 6 * 3600
# Used only if Microsoft's voice list can't be fetched.
FALLBACK_VOICES: dict[tuple[str, str], str] = {
    ("hi", "female"): "hi-IN-SwaraNeural",
    ("hi", "male"): "hi-IN-MadhurNeural",
    ("en", "female"): "en-IN-NeerjaNeural",
    ("en", "male"): "en-IN-PrabhatNeural",
}
_voice_cache: dict[str, Any] = {"at": 0.0, "voices": []}


async def _list_voices() -> list[dict[str, Any]]:
    import edge_tts

    if _voice_cache["voices"] and time.monotonic() - _voice_cache["at"] < VOICE_LIST_TTL_SECONDS:
        return _voice_cache["voices"]
    try:
        _voice_cache["voices"] = await edge_tts.list_voices()
        _voice_cache["at"] = time.monotonic()
    except Exception as exc:
        logger.warning("Could not fetch the neural voice list: %s", exc)
    return _voice_cache["voices"]


def pick_voice(voices: list[dict[str, Any]], lang: str, gender: str) -> str | None:
    """Best neural voice for a language code: Indian locale first (also for English), then the requested gender."""
    lang, gender = lang.lower()[:2], gender.lower()
    candidates = [v for v in voices if str(v.get("Locale", "")).lower().startswith(lang + "-")]
    if not candidates:
        return FALLBACK_VOICES.get((lang, gender))
    pool = [v for v in candidates if str(v["Locale"]).lower().endswith("-in")] or candidates
    pool = [v for v in pool if str(v.get("Gender", "")).lower() == gender] or pool
    # Plain voices first; "Expressive"/"Multilingual" variants are previews with odd behaviour.
    pool.sort(key=lambda v: (any(tag in v["ShortName"] for tag in ("Expressive", "Multilingual")), v["ShortName"]))
    return str(pool[0]["ShortName"])


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TTS_CHARS)
    lang: str = Field(default="en", min_length=2, max_length=8, pattern=r"^[A-Za-z-]+$")
    gender: str = Field(default="female", pattern=r"^(?i:female|male)$")


async def synthesize(text: str, voice: str) -> bytes:
    """MP3 bytes of ``text`` spoken by a Microsoft neural voice."""
    import edge_tts

    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=settings.TTS_RATE).stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    if not audio:
        raise RuntimeError("the voice service returned no audio")
    return bytes(audio)


def _allowed_models() -> set[str]:
    return {"tiny", "base", "small", "medium", settings.VOICE_MODEL, settings.VOICE_LIVE_MODEL}


def create_router(graph: CompiledStateGraph) -> APIRouter:
    router = APIRouter()

    @router.get("/live", response_class=HTMLResponse)
    def live_page() -> HTMLResponse:
        page = PAGE_PATH.read_text(encoding="utf-8")
        config = {
            "name": settings.ASSISTANT_NAME,
            "model": settings.VOICE_LIVE_MODEL,
            "silenceMs": 800,
        }
        page = page.replace("__ASSISTANT_NAME__", html.escape(settings.ASSISTANT_NAME))
        page = page.replace("__CONFIG__", json.dumps(config).replace("</", "<\\/"))
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    @router.post("/api/stt")
    async def speech_to_text(request: Request, model: str = "") -> JSONResponse:
        """Raw 16 kHz mono int16 PCM in, ``{text, language, wake, ms}`` out."""
        _guard(request)
        model = model if model in _allowed_models() else settings.VOICE_LIVE_MODEL
        body = await request.body()
        if len(body) > MAX_PCM_BYTES:
            raise HTTPException(status_code=413, detail="audio too long")
        if len(body) < MIN_PCM_BYTES:
            return JSONResponse({"text": "", "language": "", "wake": False, "ms": 0})

        started = time.perf_counter()
        try:
            audio = voice.pcm16_to_float32(body)
            text, language = await run_in_threadpool(voice.transcribe, audio, model)
        except voice.VoiceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Live transcription failed")
            raise HTTPException(status_code=500, detail=f"transcription failed ({type(exc).__name__})") from exc

        wake, command = voice.strip_wake_word(text)  # "Jadu, open youtube" -> "open youtube"
        return JSONResponse(
            {
                "text": command or text,
                "language": language,
                "wake": wake,
                "ms": round((time.perf_counter() - started) * 1000),
            }
        )

    @router.post("/api/live-chat")
    def live_chat(request: Request, body: ChatRequest) -> StreamingResponse:
        _guard(request)
        return StreamingResponse(
            stream_reply(graph, body.thread_id, body.text.strip()),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @router.post("/api/tts")
    async def text_to_speech(request: Request, body: TtsRequest) -> Response:
        """Speak one sentence with an Indian neural voice -> ``audio/mpeg``."""
        _guard(request)
        try:
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="Indian voices need: pip install edge-tts") from exc

        text = speakable(body.text)
        if not text:
            raise HTTPException(status_code=400, detail="nothing to speak")
        voice_name = pick_voice(await _list_voices(), body.lang, body.gender)
        if voice_name is None:
            raise HTTPException(status_code=404, detail=f"no neural voice for language '{body.lang}'")
        try:
            audio = await synthesize(text, voice_name)
        except Exception as exc:  # offline, service change, ...
            logger.warning("Neural TTS failed (%s): %s", voice_name, exc)
            raise HTTPException(status_code=503, detail=f"voice service unavailable ({type(exc).__name__})") from exc
        return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store", "X-Voice": voice_name})

    return router


def warm_up_in_background() -> None:
    """Load the speech model and the chat model now, so the first turn is fast."""

    def work() -> None:
        try:
            voice.warm_up(settings.VOICE_LIVE_MODEL)
        except Exception as exc:
            logger.warning("Speech model warm-up failed: %s", exc)
        try:
            import asyncio

            asyncio.run(_list_voices())  # so the first spoken sentence doesn't wait for the voice list
        except Exception as exc:
            logger.info("Neural voice list not preloaded: %s", exc)
        try:
            import ollama

            ollama.Client(host=settings.OLLAMA_BASE_URL, timeout=60).generate(
                model=settings.LLM_MODEL, prompt="", keep_alive=settings.LLM_KEEP_ALIVE
            )
        except Exception as exc:
            logger.warning("Chat model warm-up failed: %s", exc)

    threading.Thread(target=work, name="live-warmup", daemon=True).start()
