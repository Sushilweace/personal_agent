"""Central configuration for the personal agent.

All values can be overridden with environment variables (or a ``.env`` file placed
next to this module), e.g. ``LLM_MODEL=qwen3:4b python main.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Runtime settings, validated by Pydantic."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Identity --------------------------------------------------------
    # The assistant's name; also the wake word in hands-free voice mode.
    ASSISTANT_NAME: str = Field(default="Jadu", min_length=1)
    # Tone of the replies: "friendly" (default), "professional" or "playful".
    ASSISTANT_STYLE: Literal["friendly", "professional", "playful"] = "friendly"
    # Optional: what Jadu should call you (used now and then, not in every reply).
    USER_NAME: str = ""
    # Extra spellings/transcriptions of the name that count as the wake word
    # (comma separated). Speech recognition often writes "Jadu" differently.
    WAKE_ALIASES: str = "jaadu,jadoo,jaado,jado,jadu,जादू,जादु,जादो"

    # --- Ollama / models -------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:3b"
    EMBED_MODEL: str = "nomic-embed-text"

    # --- Generation ------------------------------------------------------
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    # Context window. 4096 keeps the KV cache small enough for a 4 GB GPU
    # alongside the ~2 GB llama3.2:3b weights.
    LLM_NUM_CTX: int = Field(default=4096, ge=512)
    LLM_NUM_PREDICT: int = Field(default=1024, ge=16)
    # Seconds Ollama keeps a model loaded after a request (-1 = forever).
    LLM_KEEP_ALIVE: int = 600
    LLM_TIMEOUT_SECONDS: float = Field(default=120.0, gt=0)

    # --- Agent loop ------------------------------------------------------
    # Approximate token budget of conversation history sent to the model.
    HISTORY_TOKEN_BUDGET: int = Field(default=2200, ge=256)
    # LangGraph recursion limit: caps agent<->tools round trips per user turn.
    MAX_GRAPH_STEPS: int = Field(default=14, ge=3)

    # --- Paths -----------------------------------------------------------
    DOCUMENTS_DIR: Path = BASE_DIR / "data" / "documents"
    CHROMA_DIR: Path = BASE_DIR / "memory"
    LOG_FILE: Path = BASE_DIR / "agent.log"

    # --- RAG -------------------------------------------------------------
    CHROMA_COLLECTION: str = "personal_notes"
    CHUNK_SIZE: int = Field(default=800, ge=100)
    CHUNK_OVERLAP: int = Field(default=120, ge=0)
    RETRIEVAL_TOP_K: int = Field(default=3, ge=1, le=20)
    EMBED_BATCH_SIZE: int = Field(default=16, ge=1)

    # --- Web tools -------------------------------------------------------
    WEB_MAX_RESULTS: int = Field(default=5, ge=1, le=10)
    WEB_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0)
    # Max characters of page text returned to the model (protects the context).
    WEB_MAX_CHARS: int = Field(default=3000, ge=500)

    # --- Job alerts (Gmail, read-only) ----------------------------------------
    # Your job criteria (see data/documents/job_profile.md). Read by the job tool.
    JOB_PROFILE_FILE: Path = BASE_DIR / "data" / "documents" / "job_profile.md"
    # Google sign-in files live here (git-ignored). Never share this folder.
    SECRETS_DIR: Path = BASE_DIR / "secrets"
    # Only emails from these senders/domains are ever read (comma separated).
    JOB_ALERT_SENDERS: str = (
        "linkedin.com,naukri.com,naukrigulf.com,indeed.com,foundit.in,monsterindia.com,"
        "glassdoor.com,instahyre.com,wellfound.com,cutshort.io,hirist.tech,shine.com,timesjobs.com"
    )
    JOB_ALERT_DAYS: int = Field(default=7, ge=1, le=30)
    JOB_MAX_EMAILS: int = Field(default=30, ge=1, le=100)
    JOB_TOP_N: int = Field(default=8, ge=1, le=20)

    # --- Browser UI (web.py) ----------------------------------------------
    # Keep 127.0.0.1: the agent can open pages on this computer, so the UI must
    # not be reachable from other machines.
    WEBUI_HOST: str = "127.0.0.1"
    WEBUI_PORT: int = Field(default=7860, ge=1024, le=65535)

    # --- Running like a desktop app (launcher.py) -----------------------------
    # Open Jadu in its own app-style window (Edge/Chrome without tabs or address bar).
    OPEN_AS_APP: bool = True
    # Open the window automatically when `python web.py` / the shortcut starts.
    AUTO_OPEN: bool = True
    # Start `ollama serve` in the background if it is not running yet.
    OLLAMA_AUTOSTART: bool = True

    # --- Spoken replies in live mode (Microsoft neural voices via edge-tts) --
    # Speed of the Indian neural voices, e.g. "+0%", "+10%", "-10%".
    TTS_RATE: str = Field(default="+5%", pattern=r"^[+-]\d{1,2}%$")

    # --- Voice input (optional: faster-whisper + sounddevice) -------------
    # Multilingual by default: "small" (~460 MB) understands Hindi and ~100 other
    # languages. Use "base" for less RAM/faster, "medium" for better accuracy, or
    # "base.en" for English-only speed.
    VOICE_MODEL: str = "small"
    # "" = auto-detect each time. Pin a code ("hi", "en", "bn", "ta", ...) if
    # auto-detect picks the wrong language on short commands.
    VOICE_LANGUAGE: str = ""
    # Model for the live browser conversation (/live). "base" answers in ~1.5 s on a
    # modern CPU; "small" is ~3x slower but noticeably better for Hindi.
    VOICE_LIVE_MODEL: str = "base"
    VOICE_MAX_SECONDS: float = Field(default=15.0, gt=1)
    VOICE_WAIT_SECONDS: float = Field(default=8.0, gt=1)  # give up if nobody speaks
    VOICE_SILENCE_SECONDS: float = Field(default=1.2, gt=0.2)  # pause that ends an utterance
    VOICE_SILENCE_THRESHOLD: float = Field(default=0.01, gt=0)  # RMS level counted as speech

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings: Settings = Settings()
