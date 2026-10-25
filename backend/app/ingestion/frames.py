"""Sparse frame sampling plus scene-change frames, with near-duplicate removal."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger

log = get_logger("frames")

FRAME_WIDTH = 640  # downscaled; enough for CLIP and legible OCR on slides
DUP_THRESHOLD = 3.0  # mean abs gray difference (0-255) below which frames are "the same"


@dataclass(frozen=True)
class SampledFrame:
    idx: int
    timestamp: float
    path: Path
    scene_id: int | None = None


def _gray_signature(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if img is None else cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA)


def _is_duplicate(a, b) -> bool:
    return a is not None and b is not None and float(cv2.absdiff(a, b).mean()) < DUP_THRESHOLD


def _extract_at(video: Path, ts: float, dest: Path) -> bool:
    s = get_settings()
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{ts:.3f}", "-i", str(video),
           "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "4", str(dest)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=s.ffmpeg_timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return False
    return dest.exists() and dest.stat().st_size > 0


def extract_ocr_image(video: Path, ts: float, dest: Path) -> bool:
    """Temporary near-native-resolution frame for OCR (stored frames are downscaled to
    FRAME_WIDTH, which loses small UI text). Caller deletes `dest` afterwards."""
    s = get_settings()
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{ts:.3f}", "-i", str(video), "-frames:v", "1",
           "-vf", f"scale='min(iw,{s.ocr_max_width})':-2", "-q:v", "2", str(dest)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=s.ffmpeg_timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return False
    return dest.exists() and dest.stat().st_size > 0


def scene_start_times(video: Path) -> list[float]:
    """Scene boundaries via PySceneDetect. Optional: any failure just means no extras."""
    s = get_settings()
    if not s.scene_detection_enabled:
        return []
    try:
        from scenedetect import ContentDetector, detect

        scenes = detect(str(video), ContentDetector(threshold=27.0), show_progress=False)
        return [round(start.get_seconds(), 2) for start, _end in scenes[1:]]
    except Exception as exc:
        log.warning("scene_detection_failed", error=type(exc).__name__)
        return []


def sample_frames(video: Path, duration: float, out_dir: Path) -> list[SampledFrame]:
    """Interval frames (centre of each window) merged with scene starts, sorted by time,
    near-duplicates dropped. Files are named frame_<idx>.jpg inside out_dir."""
    s = get_settings()
    interval = s.frame_sample_interval_seconds
    times = {round(min(t, max(duration - 0.1, 0.0)), 2) for t in _interval_times(duration, interval)}
    scenes = {t for t in scene_start_times(video) if t < duration}
    times |= scenes
    if not times:
        raise AppError(422, "no_frames", "Video is too short to sample frames.")

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[SampledFrame] = []
    last_sig = None
    scene_no = -1
    for ts in sorted(times):
        if ts in scenes:
            scene_no += 1
        tmp = out_dir / f"_cand_{len(kept)}.jpg"
        if not _extract_at(video, ts, tmp):
            tmp.unlink(missing_ok=True)
            continue
        sig = _gray_signature(tmp)
        if _is_duplicate(last_sig, sig) and ts not in scenes:
            tmp.unlink(missing_ok=True)
            continue
        final = out_dir / f"frame_{len(kept):05d}.jpg"
        tmp.replace(final)
        kept.append(SampledFrame(len(kept), ts, final, scene_no if scene_no >= 0 else None))
        last_sig = sig
    if not kept:
        raise AppError(422, "no_frames", "No frames could be extracted.")
    log.info("frames_sampled", count=len(kept), candidates=len(times))
    return kept


def _interval_times(duration: float, interval: float):
    t = interval / 2
    while t < duration:
        yield t
        t += interval
    if duration < interval:
        yield duration / 2
