"""Local faster-whisper transcription. The model loads once per process."""

import threading
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger

log = get_logger("transcribe")

_model = None
_lock = threading.Lock()


@dataclass(frozen=True)
class Segment:
    idx: int
    start: float
    end: float
    text: str


def _get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel

            s = get_settings()
            log.info("whisper_loading", model=s.whisper_model, device=s.whisper_device)
            _model = WhisperModel(
                s.whisper_model, device=s.whisper_device, compute_type=s.whisper_compute_type
            )
        return _model


def transcribe(wav: Path) -> list[Segment]:
    """Timestamped segments in seconds on the original media timeline. VAD drops silence,
    so a silent or music-only track yields an empty list rather than hallucinated text."""
    try:
        raw, _info = _get_model().transcribe(
            str(wav), vad_filter=True, beam_size=1, condition_on_previous_text=False
        )
        segments = [
            Segment(i, round(seg.start, 2), round(seg.end, 2), seg.text.strip())
            for i, seg in enumerate(raw)
            if seg.text.strip()
        ]
    except Exception as exc:  # model download/decoding failures: report a safe message
        log.warning("transcribe_failed", error=type(exc).__name__)
        raise AppError(500, "transcription_failed", "Transcription failed.") from exc
    # Re-index after filtering so idx stays contiguous.
    return [Segment(i, s.start, s.end, s.text) for i, s in enumerate(segments)]
