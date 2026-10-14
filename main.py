"""Interactive terminal runner for the personal agent.

Commands: ``exit`` / ``quit``, ``clear`` (new conversation), ``reindex``, ``voice``, ``help``.
"""

from __future__ import annotations

import logging
import sys
import uuid

import ollama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

import voice
from config import settings
from graph import build_graph
from rag import index_documents, indexed_chunk_count, is_index_empty

console = Console()
logger = logging.getLogger("personal_agent")

EXIT_COMMANDS: frozenset[str] = frozenset({"exit", "quit"})
HELP_TEXT: str = (
    "[bold]Commands[/bold]\n"
    "  [cyan]exit[/cyan]     leave the agent\n"
    "  [cyan]clear[/cyan]    forget the conversation and start a new session\n"
    "  [cyan]reindex[/cyan]  rebuild the document index from data/documents/\n"
    "  [cyan]voice[/cyan]    speak one command (e.g. \"open youtube\") instead of typing\n"
    "  [cyan]listen[/cyan]   hands-free: act only when you say the assistant's name (\"Jadu, open youtube\")\n"
    "  [cyan]help[/cyan]     show this message"
)


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def configure_logging() -> None:
    """Send logs to a file so tracebacks never clutter the chat UI."""
    handler = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)


def _normalize_model_name(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def check_ollama() -> bool:
    """Verify the Ollama server is reachable and both models are installed."""
    try:
        client = ollama.Client(host=settings.OLLAMA_BASE_URL, timeout=5)
        installed = {_normalize_model_name(m.model or "") for m in client.list().models}
    except Exception as exc:
        console.print(
            f"[bold red]Cannot reach Ollama at {escape(settings.OLLAMA_BASE_URL)}[/bold red] "
            f"({escape(type(exc).__name__)}).\nStart it with [cyan]ollama serve[/cyan] and try again."
        )
        return False

    ok = True
    for model in (settings.LLM_MODEL, settings.EMBED_MODEL):
        if _normalize_model_name(model) not in installed:
            console.print(f"[bold red]Model '{escape(model)}' is not installed.[/bold red] Run: [cyan]ollama pull {escape(model)}[/cyan]")
            ok = False
    return ok


def build_index(reset: bool) -> None:
    """Index documents with a progress bar and report the outcome."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Embedding document chunks", total=None)

            def on_progress(done: int, total: int) -> None:
                progress.update(task, total=total, completed=done)

            stats = index_documents(reset=reset, progress=on_progress)
    except KeyboardInterrupt:
        console.print("\n[yellow]Indexing cancelled. Type [cyan]reindex[/cyan] to resume.[/yellow]")
        return
    except Exception as exc:
        logger.exception("Indexing failed")
        console.print(f"[bold red]Indexing failed:[/bold red] {escape(str(exc))}")
        return

    for problem in stats.problems:
        console.print(f"[yellow]Skipped[/yellow] {escape(problem)}")
    if stats.chunks == 0:
        console.print(
            f"[yellow]No documents to index.[/yellow] Put .txt, .md or .pdf files in "
            f"[cyan]{escape(str(settings.DOCUMENTS_DIR))}[/cyan] and type [cyan]reindex[/cyan]."
        )
    else:
        console.print(f"[green]Indexed {stats.chunks} chunks from {stats.files} file(s).[/green]")


# --------------------------------------------------------------------------- #
# Conversation
# --------------------------------------------------------------------------- #
def new_thread_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"


def _message_text(message: BaseMessage) -> str:
    """Plain text of a message whose content may be a string or content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _show_update(node: str, update: dict[str, list[BaseMessage]]) -> str | None:
    """Render one graph update; return the final answer text if this was it."""
    final: str | None = None
    for message in update.get("messages", []):
        if node == "agent" and isinstance(message, AIMessage):
            if message.tool_calls:
                for call in message.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    console.print(f"[dim]🔧 {escape(call['name'])}({escape(_shorten(args, 100))})[/dim]")
            else:
                final = _message_text(message)
        elif isinstance(message, ToolMessage):
            failed = message.status == "error" or _message_text(message).startswith("Error")
            style = "red" if failed else "dim"
            console.print(f"[{style}]   ↳ {escape(_shorten(_message_text(message), 140))}[/{style}]")
    return final


def run_turn(graph: CompiledStateGraph, thread_id: str, user_text: str) -> None:
    """Send one user message through the graph and print the reply."""
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.MAX_GRAPH_STEPS,
    }
    answer: str | None = None
    try:
        with console.status("[cyan]Thinking…[/cyan]", spinner="dots"):
            for chunk in graph.stream(
                {"messages": [HumanMessage(content=user_text)]},
                config,
                stream_mode="updates",
            ):
                for node, update in chunk.items():
                    if isinstance(update, dict):
                        answer = _show_update(node, update) or answer
    except GraphRecursionError:
        console.print("[yellow]The agent hit its step limit without finishing. Try a simpler or more specific request.[/yellow]")
        return
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return
    except Exception as exc:
        logger.exception("Agent run failed")
        console.print(f"[bold red]Agent error:[/bold red] {escape(type(exc).__name__)}: {escape(str(exc))} (details in {escape(str(settings.LOG_FILE))})")
        return

    if answer:
        console.print(Panel(Markdown(answer), title=escape(settings.ASSISTANT_NAME), title_align="left", border_style="green"))
    else:
        console.print("[yellow]The agent returned no answer.[/yellow]")


STOP_PHRASES: frozenset[str] = frozenset(
    {"stop", "exit", "quit", "bye", "goodbye", "stop listening", "बंद", "बंद करो", "band karo", "रुको", "ruko"}
)


def capture_speech(status: str) -> tuple[str, str] | None:
    """Record one utterance and transcribe it.

    Returns ``(text, language)``, or ``None`` if nobody spoke / nothing was
    understood. Raises ``voice.VoiceError`` when voice input can't work at all.
    """
    with console.status(status, spinner="dots"):
        audio = voice.record_utterance()
    if audio is None:
        return None
    with console.status("[cyan]Transcribing… (first use downloads the speech model, ~460 MB)[/cyan]", spinner="dots"):
        text, language = voice.transcribe(audio)
    return (text, language) if text else None


def _heard(text: str, language: str) -> None:
    label = f" ({escape(language)})" if language else ""
    console.print(f"[green]🎤 Heard{label}:[/green] {escape(text)}")


def listen_for_command() -> str | None:
    """One-shot voice input: returns the spoken command, or ``None`` on failure.

    The assistant's name is optional here ("Jadu, open youtube" and plain
    "open youtube" both work); if it is present it is removed.
    """
    try:
        heard = capture_speech("[cyan]🎤 Listening… speak now[/cyan]")
    except voice.VoiceError as exc:
        console.print(f"[bold red]Voice input unavailable:[/bold red] {escape(str(exc))}")
        return None
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return None
    except Exception as exc:
        logger.exception("Voice input failed")
        console.print(f"[bold red]Voice input failed:[/bold red] {escape(type(exc).__name__)}: {escape(str(exc))}")
        return None

    if heard is None:
        console.print("[yellow]I didn't catch that. Type [cyan]voice[/cyan] to try again.[/yellow]")
        return None
    text, language = heard
    _heard(text, language)
    _found, command = voice.strip_wake_word(text)
    if not command:
        console.print(f"[yellow]Say your request after the name, e.g. “{escape(settings.ASSISTANT_NAME)}, open youtube”.[/yellow]")
        return None
    return command


def hands_free(graph: CompiledStateGraph, thread_id: str) -> None:
    """Listen continuously and act only on speech addressed to the assistant by name.

    "Jadu, open youtube" runs the request. Saying just "Jadu" makes it wait for
    the request in the next utterance. "Jadu stop" (or Ctrl+C) leaves the mode.
    """
    name = settings.ASSISTANT_NAME
    console.print(
        f"[bold cyan]🎤 Hands-free mode.[/bold cyan] Say “[bold]{escape(name)}[/bold], <request>”. "
        f"Say “{escape(name)} stop” or press Ctrl+C to leave."
    )
    awaiting_request = False
    try:
        while True:
            label = "[cyan]Yes? I'm listening…[/cyan]" if awaiting_request else f"[dim]Waiting for “{escape(name)}”…[/dim]"
            try:
                heard = capture_speech(label)
            except voice.VoiceError as exc:
                console.print(f"[bold red]Voice input unavailable:[/bold red] {escape(str(exc))}")
                return
            except Exception as exc:
                logger.exception("Voice input failed")
                console.print(f"[bold red]Voice input failed:[/bold red] {escape(type(exc).__name__)}: {escape(str(exc))}")
                return
            if heard is None:
                awaiting_request = False  # silence or noise: go back to waiting for the name
                continue

            text, language = heard
            found, command = voice.strip_wake_word(text)
            if not found and not awaiting_request:
                console.print(f"[dim]ignored (no “{escape(name)}”): {escape(_shorten(text, 80))}[/dim]")
                continue
            if found and not command:
                awaiting_request = True
                console.print(f"[green]🎤 {escape(name)}:[/green] yes?")
                continue

            _heard(text, language)
            awaiting_request = False
            if command.casefold().strip(" .!।") in STOP_PHRASES:
                console.print("[dim]Leaving hands-free mode.[/dim]")
                return
            run_turn(graph, thread_id, command)
    except KeyboardInterrupt:
        console.print("\n[dim]Leaving hands-free mode.[/dim]")


def print_banner(thread_id: str) -> None:
    console.print(
        Panel.fit(
            f"[bold]{escape(settings.ASSISTANT_NAME)}[/bold] - personal AI agent\n"
            f"model: [cyan]{escape(settings.LLM_MODEL)}[/cyan]   embeddings: [cyan]{escape(settings.EMBED_MODEL)}[/cyan]\n"
            f"knowledge base: [cyan]{indexed_chunk_count()}[/cyan] chunks   session: [cyan]{thread_id}[/cyan]\n"
            f"Type [cyan]help[/cyan] for commands.",
            border_style="blue",
        )
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # avoid UnicodeEncodeError on legacy Windows consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    configure_logging()

    if not check_ollama():
        return 1

    try:
        if is_index_empty():
            console.print("[cyan]Knowledge base is empty - indexing documents…[/cyan]")
            build_index(reset=False)
    except Exception as exc:
        logger.exception("Could not open the vector store")
        console.print(f"[bold red]Vector store error:[/bold red] {escape(str(exc))}")
        return 1

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    thread_id = new_thread_id()
    print_banner(thread_id)

    while True:
        try:
            raw = console.input("\n[bold green]You[/bold green] › ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            return 0

        text = raw.strip()
        if not text:
            continue
        command = text.lower().lstrip("/")

        if command in EXIT_COMMANDS:
            console.print("[dim]Goodbye![/dim]")
            return 0
        if command == "help":
            console.print(HELP_TEXT)
        elif command == "clear":
            checkpointer.delete_thread(thread_id)
            thread_id = new_thread_id()
            if console.is_terminal:
                console.clear()
            console.print(f"[green]Conversation cleared.[/green] New session: [cyan]{thread_id}[/cyan]")
        elif command == "reindex":
            console.print("[cyan]Rebuilding the index…[/cyan]")
            build_index(reset=True)
        elif command == "voice":
            spoken = listen_for_command()
            if spoken:
                run_turn(graph, thread_id, spoken)
        elif command == "listen":
            hands_free(graph, thread_id)
        else:
            run_turn(graph, thread_id, text)


if __name__ == "__main__":
    sys.exit(main())
