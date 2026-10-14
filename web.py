"""Browser UI for the personal agent.

Run with ``python web.py``. It serves the same LangGraph agent, tools, voice
transcription and assistant name as the terminal app (``main.py``), on
``http://127.0.0.1:7860`` by default:

* ``/live``  hands-free voice conversation (automatic start/stop, spoken replies; see ``live.py``)
* ``/``      Gradio text chat with a microphone button

Each browser tab gets its own conversation.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator
from typing import Any

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

import connections
import launcher
import live
import voice
from config import settings
from graph import build_graph
from main import _message_text, _shorten, build_index, check_ollama, configure_logging
from rag import index_documents, is_index_empty

logger = logging.getLogger("personal_agent.web")

ChatHistory = list[dict[str, Any]]


def new_thread_id() -> str:
    return f"web-{uuid.uuid4().hex[:8]}"


def _tool_bubble(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """A collapsible 'tool running' chat bubble."""
    arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return {
        "role": "assistant",
        "content": "",
        "metadata": {"title": f"🔧 {name}({_shorten(arg_text, 100)})", "status": "pending"},
    }


def _finish_pending(history: ChatHistory) -> None:
    for item in history:
        meta = item.get("metadata")
        if isinstance(meta, dict) and meta.get("status") == "pending":
            meta["status"] = "done"


def create_app(graph: CompiledStateGraph, checkpointer: MemorySaver) -> gr.Blocks:
    """Build the Gradio interface around an existing compiled graph."""
    name = settings.ASSISTANT_NAME

    def chat(message: str, history: ChatHistory | None, thread_id: str) -> Iterator[tuple[ChatHistory, str]]:
        history = list(history or [])
        message = (message or "").strip()
        if not message:
            yield history, ""
            return

        history.append({"role": "user", "content": message})
        yield list(history), ""

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.MAX_GRAPH_STEPS}
        answer: str | None = None
        try:
            for chunk in graph.stream({"messages": [HumanMessage(content=message)]}, config, stream_mode="updates"):
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    for msg in update.get("messages", []):
                        if node == "agent" and isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for call in msg.tool_calls:
                                    history.append(_tool_bubble(call["name"], call["args"]))
                            else:
                                answer = _message_text(msg)
                        elif isinstance(msg, ToolMessage):
                            pending = [
                                m for m in history
                                if isinstance(m.get("metadata"), dict) and m["metadata"].get("status") == "pending"
                            ]
                            if pending:
                                pending[0]["content"] = _shorten(_message_text(msg), 600)
                                pending[0]["metadata"]["status"] = "done"
                        yield list(history), ""
        except GraphRecursionError:
            answer = "I hit my step limit without finishing. Try a simpler or more specific request."
        except Exception as exc:
            logger.exception("Agent run failed")
            answer = f"⚠️ Agent error: {type(exc).__name__}: {exc} (details in {settings.LOG_FILE})"

        _finish_pending(history)
        history.append({"role": "assistant", "content": answer or "I couldn't produce an answer."})
        yield list(history), ""

    def transcribe_audio(recording: tuple[int, Any] | None) -> str:
        """Turn a microphone recording ``(sample_rate, samples)`` into text for the message box."""
        if recording is None:
            return ""
        try:
            sample_rate, samples = recording
            text, language = voice.transcribe(voice.to_whisper_audio(sample_rate, samples))
        except voice.VoiceError as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:
            logger.exception("Browser transcription failed")
            raise gr.Error(f"Could not transcribe the audio ({type(exc).__name__}: {exc})") from exc
        if not text:
            gr.Warning("I couldn't make out any words. Please try again.")
            return ""
        gr.Info(f"Heard ({language or '?'}): {text}")
        _found, command = voice.strip_wake_word(text)  # "Jadu, open youtube" -> "open youtube"
        return command or text

    def new_chat(thread_id: str) -> tuple[ChatHistory, str]:
        if thread_id:
            checkpointer.delete_thread(thread_id)
        return [], new_thread_id()

    def reindex() -> None:
        try:
            stats = index_documents(reset=True)
        except Exception as exc:
            logger.exception("Indexing failed")
            raise gr.Error(f"Indexing failed: {exc}") from exc
        gr.Info(f"Indexed {stats.chunks} chunks from {stats.files} file(s).")
        for problem in stats.problems:
            gr.Warning(f"Skipped {problem}")

    with gr.Blocks(title=name) as demo:
        gr.Markdown(
            f"# 🪄 {name}\n"
            f"Your local personal AI agent (`{settings.LLM_MODEL}`). Type below, or press the microphone, "
            f"speak (any language, e.g. “{name}, open youtube” or Hindi), and stop recording to send.\n\n"
            f"**🎙️ Want a hands-free voice conversation (no buttons, it talks back)? "
            f"[Open live voice mode](/live)** · **[🔗 Connections (Gmail, LinkedIn, Naukri)](/connections)**"
        )
        thread = gr.State("")
        chatbot = gr.Chatbot(height=500, label=name, placeholder=f"Ask {name} anything…")
        with gr.Row():
            box = gr.Textbox(
                show_label=False, placeholder=f"Message {name}…", scale=5, autofocus=True, lines=1, max_lines=6
            )
            mic = gr.Audio(sources=["microphone"], type="numpy", show_label=False, scale=3)
        with gr.Row():
            new_btn = gr.Button("🗑️ New chat", scale=1)
            index_btn = gr.Button("🔄 Re-index documents", scale=1)

        demo.load(new_thread_id, None, thread)
        box.submit(chat, [box, chatbot, thread], [chatbot, box])
        mic.stop_recording(transcribe_audio, mic, box).success(
            chat, [box, chatbot, thread], [chatbot, box]
        ).then(lambda: None, None, mic)
        new_btn.click(new_chat, thread, [chatbot, thread])
        index_btn.click(reindex, None, None)

    return demo


def create_server(graph: CompiledStateGraph, checkpointer: MemorySaver) -> FastAPI:
    """One server for everything: ``/live`` (voice conversation, ``/api/*``) and ``/`` (text chat)."""
    app = FastAPI(title=settings.ASSISTANT_NAME, docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(live.create_router(graph))
    app.include_router(connections.create_router())

    @app.post("/api/quit")
    def quit_app(request: Request) -> dict[str, bool]:
        """Stop the server (the "Quit" button). Same localhost/header guard as the other APIs."""
        live._guard(request)
        server = getattr(request.app.state, "server", None)
        if server is not None:
            server.should_exit = True
        return {"ok": server is not None}

    return gr.mount_gradio_app(app, create_app(graph, checkpointer), path="/", theme=gr.themes.Soft())


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    configure_logging()

    base = f"http://{settings.WEBUI_HOST}:{settings.WEBUI_PORT}"
    if launcher.port_open(settings.WEBUI_HOST, settings.WEBUI_PORT):
        print(f"{settings.ASSISTANT_NAME} is already running at {base}; opening it.")
        launcher.open_window(f"{base}/live")
        return 0

    launcher.ensure_ollama()
    if not check_ollama():
        return 1
    try:
        if is_index_empty():
            build_index(reset=False)
    except Exception as exc:
        logger.exception("Could not open the vector store")
        print(f"Vector store error: {exc}")
        return 1

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    app = create_server(graph, checkpointer)
    live.warm_up_in_background()  # loads the speech + chat models while the page opens

    print(
        f"{settings.ASSISTANT_NAME} is running:\n  live voice  {base}/live\n  connections {base}/connections\n"
        f"  text chat   {base}/\n(Use the Quit button in the app, or press Ctrl+C here, to stop)"
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=settings.WEBUI_HOST,
            port=settings.WEBUI_PORT,
            log_level="warning",
            timeout_graceful_shutdown=3,  # open chat streams must not block "Quit"
        )
    )
    app.state.server = server
    if settings.AUTO_OPEN:
        launcher.open_when_ready(f"{base}/live")
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
