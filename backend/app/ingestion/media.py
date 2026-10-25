"""ffprobe/ffmpeg wrappers. Every call uses an argument list (no shell) and a timeout."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger

log = get_logger("media")

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float
    width: int | None
    height: int | None
    frame_rate: float | None
    has_audio: bool


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise AppError(422, "media_timeout", "Media processing timed out.") from exc
    except FileNotFoundError as exc:
        raise AppError(500, "tool_missing", f"{cmd[0]} is not installed.") from exc


def _parse_rate(raw: str | None) -> float | None:
    if not raw or "/" not in raw:
        return None
    num, den = raw.split("/", 1)
    try:
        n, d = float(num), float(den)
    except ValueError:
        return None
    return round(n / d, 3) if d else None


def probe(path: Path) -> MediaInfo:
    """Read duration/streams. Raises AppError(422) for anything that is not playable video."""
    out = _run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
         "-show_streams", str(path)],
        timeout=30,
    )
    if out.returncode != 0:
        raise AppError(422, "invalid_media", "File is not a readable video.")
    try:
        data = json.loads(out.stdout)
    except ValueError as exc:
        raise AppError(422, "invalid_media", "File is not a readable video.") from exc

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise AppError(422, "no_video_stream", "File contains no video stream.")
    try:
        duration = float(data.get("format", {}).get("duration") or video.get("duration"))
    except (TypeError, ValueError) as exc:
        raise AppError(422, "invalid_media", "Could not determine video duration.") from exc

    return MediaInfo(
        duration_seconds=round(duration, 3),
        width=video.get("width"),
        height=video.get("height"),
        frame_rate=_parse_rate(video.get("avg_frame_rate")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def enforce_limits(info: MediaInfo) -> None:
    s = get_settings()
    if info.duration_seconds > s.max_video_duration_seconds:
        raise AppError(
            413, "video_too_long",
            f"Video is longer than the {s.max_video_duration_seconds // 60}-minute limit.",
        )


def extract_audio(video: Path, dest_wav: Path) -> Path:
    """16 kHz mono PCM, the format Whisper wants."""
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    out = _run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1",
         "-ar", "16000", "-c:a", "pcm_s16le", str(dest_wav)],
        timeout=get_settings().ffmpeg_timeout_seconds,
    )
    if out.returncode != 0 or not dest_wav.exists():
        log.warning("audio_extract_failed", rc=out.returncode)
        raise AppError(422, "audio_extract_failed", "Could not extract audio.")
    return dest_wav
