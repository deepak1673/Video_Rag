"""Per-video ingestion. The audio and visual branches fail independently: one branch
failing leaves the video `partial` (still searchable) instead of `failed`.

A branch only succeeds after its vectors are written and the point counts verified, so
`ready` never precedes durable indexing.
"""

import shutil
import uuid
from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.models import (
    BranchStatus,
    Frame,
    ProcessingEvent,
    TranscriptChunk,
    TranscriptSegment,
    Video,
    VideoStatus,
    utcnow,
)
from app.db.session import session_scope
from app.ingestion import frames as frames_mod
from app.ingestion import media, ocr, transcribe
from app.ingestion.chunking import build_chunks
from app.retrieval import index
from app.retrieval.embeddings import embed_images, embed_texts

log = get_logger("pipeline")

_FRAME_NS = uuid.UUID("5f0c1d52-8d0e-4c6e-9a39-6a1f0f6c2b11")


def frame_id(video_id: uuid.UUID, idx: int) -> uuid.UUID:
    """Deterministic so re-processing yields the same ids (idempotent indexing)."""
    return uuid.uuid5(_FRAME_NS, f"{video_id}:{idx}")


def _safe_message(exc: Exception) -> tuple[str, str]:
    """(code, message) that is safe to show users: never raw exception text."""
    if isinstance(exc, AppError):
        return exc.code, exc.message
    return "internal_error", "An unexpected error occurred."


def _event(db: Session, video: Video, branch: str, stage: str, status: str,
           exc: Exception | None = None) -> None:
    code, msg = _safe_message(exc) if exc else (None, None)
    db.add(ProcessingEvent(
        video_id=video.id, branch=branch, stage=stage, status=status,
        finished_at=utcnow() if status != BranchStatus.running.value else None,
        error_code=code, safe_message=msg,
    ))


def _set(db: Session, video: Video, **fields) -> None:
    for k, v in fields.items():
        setattr(video, k, v)
    db.commit()


def _audio_branch(db: Session, video: Video) -> None:
    s = get_settings()
    _set(db, video, audio_status=BranchStatus.running.value, current_stage="audio:extract",
         audio_error=None)
    if not video.has_audio:
        _set(db, video, audio_status=BranchStatus.skipped.value)
        _event(db, video, "audio", "extract", BranchStatus.skipped.value)
        return
    wav = s.tmp_dir / f"{video.id}.wav"
    try:
        media.extract_audio(s.videos_dir / video.stored_name, wav)
        _set(db, video, current_stage="audio:transcribe", progress=0.3)
        segments = transcribe.transcribe(wav)
    finally:
        wav.unlink(missing_ok=True)
    db.execute(delete(TranscriptChunk).where(TranscriptChunk.video_id == video.id))
    db.execute(delete(TranscriptSegment).where(TranscriptSegment.video_id == video.id))
    rows = [
        TranscriptSegment(video_id=video.id, idx=g.idx, start_seconds=g.start,
                          end_seconds=g.end, text=g.text)
        for g in segments
    ]
    db.add_all(rows)
    _set(db, video, current_stage="audio:index", progress=0.45)

    chunks = build_chunks(video.id, rows)
    db.add_all(
        TranscriptChunk(id=c.id, video_id=video.id, idx=c.idx, start_seconds=c.start,
                        end_seconds=c.end, text=c.text, segment_idxs=c.segment_idxs)
        for c in chunks
    )
    index.ensure_collections()
    index.delete_video(video.id, (index.TRANSCRIPT,))  # drop stale vectors from earlier runs
    vectors = embed_texts([c.text for c in chunks])
    index.upsert(index.TRANSCRIPT, [
        (c.id, vec, {"video_id": str(video.id), "evidence_id": str(c.id), "start": c.start,
                     "end": c.end, "text": c.text})
        for c, vec in zip(chunks, vectors, strict=True)
    ])
    if index.count_for_video(index.TRANSCRIPT, video.id) != len(chunks):
        raise AppError(500, "index_verify_failed", "Transcript index verification failed.")
    _set(db, video, audio_status=BranchStatus.succeeded.value, progress=0.55)
    _event(db, video, "audio", "transcribe+index", BranchStatus.succeeded.value)


def _ocr_frame(source, frame: frames_mod.SampledFrame) -> str | None:
    """OCR a temporary high-resolution copy of the frame; fall back to the stored one."""
    hires = frame.path.with_name(f"_ocr_{frame.idx}.jpg")
    try:
        path = hires if frames_mod.extract_ocr_image(source, frame.timestamp, hires) else frame.path
        return ocr.ocr_image(path)
    finally:
        hires.unlink(missing_ok=True)


def _visual_branch(db: Session, video: Video) -> None:
    s = get_settings()
    _set(db, video, visual_status=BranchStatus.running.value, current_stage="visual:frames",
         visual_error=None)
    out_dir = s.frames_dir / str(video.id)
    sampled = frames_mod.sample_frames(
        s.videos_dir / video.stored_name, video.duration_seconds or 0.0, out_dir
    )
    _set(db, video, current_stage="visual:ocr", progress=0.7)
    db.execute(delete(Frame).where(Frame.video_id == video.id))
    source = s.videos_dir / video.stored_name
    rows = [
        Frame(id=frame_id(video.id, f.idx), video_id=video.id, idx=f.idx,
              timestamp_seconds=f.timestamp, image_name=f.path.name,
              ocr_text=_ocr_frame(source, f), scene_id=f.scene_id)
        for f in sampled
    ]
    db.add_all(rows)
    _set(db, video, current_stage="visual:index", progress=0.8)

    index.ensure_collections()
    index.delete_video(video.id, (index.OCR, index.VISUAL))
    with_text = [r for r in rows if r.ocr_text]
    index.upsert(index.OCR, [
        (r.id, vec, {"video_id": str(video.id), "evidence_id": str(r.id),
                     "start": r.timestamp_seconds, "end": r.timestamp_seconds, "text": r.ocr_text})
        for r, vec in zip(with_text, embed_texts([r.ocr_text for r in with_text]), strict=True)
    ])
    index.upsert(index.VISUAL, [
        (r.id, vec, {"video_id": str(video.id), "evidence_id": str(r.id),
                     "start": r.timestamp_seconds, "end": r.timestamp_seconds})
        for r, vec in zip(rows, embed_images([f.path for f in sampled]), strict=True)
    ])
    if (index.count_for_video(index.OCR, video.id) != len(with_text)
            or index.count_for_video(index.VISUAL, video.id) != len(rows)):
        raise AppError(500, "index_verify_failed", "Frame index verification failed.")
    _set(db, video, visual_status=BranchStatus.succeeded.value, progress=0.95)
    _event(db, video, "visual", "frames+ocr+index", BranchStatus.succeeded.value)


def _run_branch(db: Session, video: Video, branch: str, fn) -> bool:
    try:
        fn(db, video)
        return True
    except Exception as exc:
        db.rollback()
        code, msg = _safe_message(exc)
        log.warning("branch_failed", video_id=str(video.id), branch=branch, code=code,
                    error=type(exc).__name__)
        _set(db, video, **{f"{branch}_status": BranchStatus.failed.value,
                           f"{branch}_error": msg})
        _event(db, video, branch, "run", BranchStatus.failed.value, exc)
        db.commit()
        return False


def process_video(video_id: uuid.UUID) -> None:
    with session_scope() as db:
        video = db.get(Video, video_id)
        if video is None or video.status == VideoStatus.deleting.value:
            return
        _set(db, video, status=VideoStatus.processing.value, progress=0.05,
             error_summary=None, current_stage="starting")
        if not (get_settings().videos_dir / video.stored_name).exists():
            _set(db, video, status=VideoStatus.failed.value,
                 error_summary="Source video file is missing.")
            return

        audio_ok = _run_branch(db, video, "audio", _audio_branch)
        visual_ok = _run_branch(db, video, "visual", _visual_branch)

        if audio_ok and visual_ok:
            final = VideoStatus.ready
        elif audio_ok or visual_ok:
            final = VideoStatus.partial
        else:
            final = VideoStatus.failed
        summary = None if final is VideoStatus.ready else "One or more processing steps failed."
        _set(db, video, status=final.value, progress=1.0 if final is not VideoStatus.failed else video.progress,
             current_stage=None, error_summary=summary, processed_at=datetime.now(UTC))
        log.info("video_processed", video_id=str(video_id), status=final.value)


def cleanup_artifacts(video_id: uuid.UUID) -> None:
    s = get_settings()
    shutil.rmtree(s.frames_dir / str(video_id), ignore_errors=True)
