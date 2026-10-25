"""Tesseract OCR with a per-frame timeout. Missing Tesseract degrades to no OCR.

Words below a confidence cutoff are dropped: on dense screen recordings Tesseract emits
a lot of junk tokens, and junk poisons both embeddings and LLM context."""

import re
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.system import find_tesseract

log = get_logger("ocr")

_MIN_CHARS = 4  # shorter than this is almost always noise
_WS = re.compile(r"\s+")
_configured = False


def _configure() -> bool:
    global _configured
    import pytesseract

    exe = find_tesseract()
    if exe is None:
        return False
    if not _configured:
        pytesseract.pytesseract.tesseract_cmd = exe
        _configured = True
    return True


def clean_words(words: list[str], confidences: list[float], min_conf: float) -> str:
    """Keep confident, non-trivial tokens in reading order; collapse whitespace."""
    kept = [
        w.strip() for w, c in zip(words, confidences, strict=True)
        if w.strip() and c >= min_conf and sum(ch.isalnum() for ch in w) >= 2
    ]
    return _WS.sub(" ", " ".join(kept)).strip()


def ocr_image(path: Path) -> str | None:
    """Cleaned text from one frame, or None if nothing legible / OCR unavailable.
    Pass a full-resolution image: downscaled frames lose small UI text."""
    s = get_settings()
    if not s.ocr_enabled or not _configure():
        return None
    import pytesseract
    from PIL import Image

    try:
        with Image.open(path) as img:
            # psm 11 (sparse text) suits slides, UI chrome and tables better than page layout.
            data = pytesseract.image_to_data(
                img.convert("L"), config="--psm 11", timeout=s.ocr_timeout_seconds,
                output_type=pytesseract.Output.DICT,
            )
    except Exception as exc:  # timeout, corrupt image, tesseract crash
        log.warning("ocr_failed", error=type(exc).__name__)
        return None
    text = clean_words(data["text"], [float(c) for c in data["conf"]], s.ocr_min_confidence)
    if len(text) < _MIN_CHARS or sum(c.isalnum() for c in text) < _MIN_CHARS:
        return None
    return text
