"""Local voice input: microphone capture (sounddevice) + faster-whisper transcription.

Everything runs offline once the Whisper model has been downloaded (first use).
Transcription runs on the CPU so the GPU stays free for the chat model.

The heavy imports are lazy: the agent works without the voice packages installed.
"""

from __future__ import annotations

import logging
import os
import threading
import unicodedata
from collections import deque
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 16_000  # Whisper's native rate
BLOCK_SECONDS: float = 0.1
PRE_ROLL_BLOCKS: int = 3  # audio kept from just before speech starts (avoids clipped first word)

_TRANSCRIBE_LOCK = threading.Lock()

INSTALL_HINT: str = "Voice input needs extra packages: pip install faster-whisper sounddevice"


class VoiceError(RuntimeError):
    """A user-presentable problem with voice input (missing package, no microphone, ...)."""


def _import_audio() -> tuple[Any, Any]:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise VoiceError(INSTALL_HINT) from exc
    return np, sd


def record_utterance() -> Any | None:
    """Record from the default microphone until the speaker pauses.

    Returns:
        A mono ``float32`` numpy array at 16 kHz, or ``None`` if nobody spoke
        within ``VOICE_WAIT_SECONDS``.

    Raises:
        VoiceError: If the packages or the microphone are unavailable.
    """
    np, sd = _import_audio()
    block = int(SAMPLE_RATE * BLOCK_SECONDS)
    max_blocks = int(settings.VOICE_MAX_SECONDS / BLOCK_SECONDS)
    silence_limit = int(settings.VOICE_SILENCE_SECONDS / BLOCK_SECONDS)
    wait_limit = int(settings.VOICE_WAIT_SECONDS / BLOCK_SECONDS)

    pre_roll: deque[Any] = deque(maxlen=PRE_ROLL_BLOCKS)
    chunks: list[Any] = []
    speaking = False
    silent_blocks = 0
    waited_blocks = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block
        ) as stream:
            while True:
                data, _overflowed = stream.read(block)
                mono = data[:, 0].copy()
                loud = float(np.sqrt(np.mean(mono**2))) > settings.VOICE_SILENCE_THRESHOLD

                if not speaking:
                    if loud:
                        speaking = True
                        chunks.extend(pre_roll)
                        chunks.append(mono)
                    else:
                        pre_roll.append(mono)
                        waited_blocks += 1
                        if waited_blocks >= wait_limit:
                            return None
                    continue

                chunks.append(mono)
                silent_blocks = 0 if loud else silent_blocks + 1
                if silent_blocks >= silence_limit or len(chunks) >= max_blocks:
                    break
    except (sd.PortAudioError, ValueError) as exc:
        raise VoiceError(f"Microphone problem: {exc}. Check your default input device.") from exc

    return np.concatenate(chunks)


@lru_cache(maxsize=4)
def _get_model(name: str) -> Any:
    """Load (once per model name) a faster-whisper model on the CPU."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise VoiceError(INSTALL_HINT) from exc
    try:
        return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=min(8, os.cpu_count() or 4))
    except Exception as exc:  # e.g. first-time download without internet
        raise VoiceError(
            f"Could not load speech model '{name}' ({type(exc).__name__}: {exc}). "
            "The first use downloads it, so an internet connection is required once."
        ) from exc


def warm_up(model_name: str | None = None) -> None:
    """Load a speech model now so the first spoken command isn't slow (downloads it on first use)."""
    _get_model(model_name or settings.VOICE_MODEL)


# --------------------------------------------------------------------------- #
# Raw audio helpers (we hand numpy arrays to Whisper; this avoids ffmpeg/PyAV decoding)
# --------------------------------------------------------------------------- #
def pcm16_to_float32(data: bytes) -> Any:
    """Little-endian 16-bit mono PCM bytes (16 kHz) -> float32 array in [-1, 1]."""
    np, _sd = _import_audio()
    usable = len(data) - (len(data) % 2)
    return np.frombuffer(data[:usable], dtype="<i2").astype("float32") / 32768.0


def to_whisper_audio(sample_rate: int, samples: Any) -> Any:
    """Convert a recorded clip (e.g. Gradio's ``(rate, ndarray)``) to 16 kHz mono float32."""
    np, _sd = _import_audio()
    audio = np.asarray(samples)
    if audio.dtype.kind in "iu":  # scale integers to [-1, 1] *before* any averaging turns them into floats
        audio = audio.astype("float32") / float(np.iinfo(audio.dtype).max + 1)
    else:
        audio = audio.astype("float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE and len(audio):
        target = max(1, int(len(audio) * SAMPLE_RATE / sample_rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, target), np.arange(len(audio)), audio).astype("float32")
    return audio


# --------------------------------------------------------------------------- #
# Wake word ("Jadu, open youtube")
# --------------------------------------------------------------------------- #
WAKE_MATCH_RATIO: float = 0.8  # fuzzy match tolerance for mis-transcribed names
WAKE_SEARCH_WORDS: int = 3  # the name must be among the first words ("hey jadu, ...")
_PUNCTUATION: str = " \t\r\n,.!?;:-–—\"'।॥"
_GREETINGS: frozenset[str] = frozenset(
    {"hey", "hi", "hello", "ok", "okay", "oi", "yo", "listen", "हे", "हेलो", "अरे", "सुनो", "ओके", "ए"}
)


def _normalize(word: str) -> str:
    """Lower-case a word and keep letters, combining marks and digits (works for Devanagari)."""
    return "".join(ch for ch in word.casefold() if unicodedata.category(ch)[0] in "LMN")


def wake_words() -> list[str]:
    """The assistant name plus its configured aliases, normalized."""
    names = [settings.ASSISTANT_NAME, *settings.WAKE_ALIASES.split(",")]
    return list(dict.fromkeys(n for n in (_normalize(x) for x in names) if n))


def _is_wake_word(word: str, wakes: list[str]) -> bool:
    norm = _normalize(word)
    return bool(norm) and any(
        norm == wake or SequenceMatcher(None, norm, wake).ratio() >= WAKE_MATCH_RATIO for wake in wakes
    )


def strip_wake_word(text: str) -> tuple[bool, str]:
    """Detect the assistant name in a transcript and remove it.

    Matches when the name is one of the first few words ("Jadu, open youtube",
    "hey Jadu open youtube") or the last word ("open youtube, Jadu").

    Returns:
        ``(found, command)`` where ``command`` is the text with the name (and
        anything before it) removed. If the name wasn't found, ``found`` is
        ``False`` and ``command`` is the original text.
    """
    words = text.split()
    wakes = wake_words()
    for index, word in enumerate(words[:WAKE_SEARCH_WORDS]):
        if not _is_wake_word(word, wakes):
            continue
        after = " ".join(words[index + 1 :]).strip(_PUNCTUATION)
        if after:
            return True, after
        # Name spoken last: "open youtube, Jadu" -> the request is what came before it
        # (minus greetings, so "hey Jadu" alone stays an empty request).
        before = [w for w in words[:index] if _normalize(w) not in _GREETINGS]
        return True, " ".join(before).strip(_PUNCTUATION)
    if len(words) > 1 and _is_wake_word(words[-1], wakes):
        return True, " ".join(words[:-1]).strip(_PUNCTUATION)
    return False, text.strip()


def transcribe(audio: Any, model_name: str | None = None) -> tuple[str, str]:
    """Transcribe a 16 kHz mono float32 array.

    Args:
        audio: 16 kHz mono float32 samples (see ``pcm16_to_float32`` / ``to_whisper_audio``).
        model_name: Whisper model to use; defaults to ``settings.VOICE_MODEL``.

    Returns:
        ``(text, language_code)``. ``text`` is empty if nothing was understood.
        The text stays in the spoken language (Hindi comes back in Devanagari);
        it is not translated.
    """
    model = _get_model(model_name or settings.VOICE_MODEL)
    with _TRANSCRIBE_LOCK:  # one transcription at a time keeps latency predictable
        segments, info = model.transcribe(
            audio,
            language=settings.VOICE_LANGUAGE or None,  # None = auto-detect
            task="transcribe",
            initial_prompt=f"{settings.ASSISTANT_NAME}.",  # nudges Whisper to spell the name correctly
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=True,  # also suppresses Whisper's "Thank you." hallucinations on silence
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, str(getattr(info, "language", "") or settings.VOICE_LANGUAGE)
