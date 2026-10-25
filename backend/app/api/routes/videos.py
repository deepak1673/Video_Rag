import shutil
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import require_access
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.models import Frame, TranscriptSegment, Video, VideoStatus
from app.db.session import get_db
from app.ingestion import media
from app.ingestion.storage import save_upload
from app.retrieval import index
from app.schemas.search import FrameOut, SegmentOut
from app.schemas.video import VideoOut, VideoStatusOut
from app.services import system

log = get_logger("videos")

router = APIRouter(prefix="/videos", tags=["videos"], dependencies=[Depends(require_access)])


def _get_or_404(db: Session, video_id: UUID) -> Video:
    video = db.get(Video, video_id)
    if video is None or video.status == VideoStatus.deleting.value:
        raise AppError(404, "video_not_found", "Video not found.")
    return video


@router.post("", response_model=VideoOut, status_code=201)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)) -> Video:
    s = get_settings()
    if not s.uploads_enabled:
        raise AppError(403, "uploads_disabled", "Uploads are disabled on this deployment.")
    if system.disk_free_gb() < s.min_free_disk_gb:
        raise AppError(507, "insufficient_storage", "Not enough free disk space for uploads.")

    path, stored_name, size = await save_upload(file)
    try:
        info = media.probe(path)
        media.enforce_limits(info)
    except BaseException:
        path.unlink(missing_ok=True)
        raise

    video = Video(
        original_filename=(file.filename or "video")[:255],
        stored_name=stored_name,
        mime_type=file.content_type or "application/octet-stream",
        size_bytes=size,
        duration_seconds=info.duration_seconds,
        width=info.width,
        height=info.height,
        frame_rate=info.frame_rate,
        has_audio=info.has_audio,
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    log.info("video_uploaded", video_id=str(video.id), size=size,
             duration=info.duration_seconds)
    return video


@router.get("", response_model=list[VideoOut])
def list_videos(db: Session = Depends(get_db)) -> list[Video]:
    stmt = (
        select(Video)
        .where(Video.status != VideoStatus.deleting.value)
        .order_by(Video.created_at.desc())
    )
    return list(db.scalars(stmt))


@router.get("/{video_id}", response_model=VideoOut)
def get_video(video_id: UUID, db: Session = Depends(get_db)) -> Video:
    return _get_or_404(db, video_id)


@router.get("/{video_id}/status", response_model=VideoStatusOut)
def get_status(video_id: UUID, db: Session = Depends(get_db)) -> Video:
    return _get_or_404(db, video_id)


@router.get("/{video_id}/transcript", response_model=list[SegmentOut])
def get_transcript(video_id: UUID, db: Session = Depends(get_db)) -> list[TranscriptSegment]:
    _get_or_404(db, video_id)
    stmt = (
        select(TranscriptSegment)
        .where(TranscriptSegment.video_id == video_id)
        .order_by(TranscriptSegment.idx)
        .limit(5000)
    )
    return list(db.scalars(stmt))


@router.get("/{video_id}/frames", response_model=list[FrameOut])
def get_frames(video_id: UUID, db: Session = Depends(get_db)) -> list[FrameOut]:
    _get_or_404(db, video_id)
    rows = db.scalars(
        select(Frame).where(Frame.video_id == video_id).order_by(Frame.idx).limit(2000)
    )
    return [
        FrameOut(
            id=f.id, idx=f.idx, timestamp_seconds=f.timestamp_seconds, ocr_text=f.ocr_text,
            scene_id=f.scene_id,
            thumbnail_url=f"/api/v1/videos/{video_id}/frames/{f.id}/thumbnail",
        )
        for f in rows
    ]


@router.get("/{video_id}/frames/{frame_id}/thumbnail")
def get_thumbnail(video_id: UUID, frame_id: UUID, db: Session = Depends(get_db)) -> FileResponse:
    _get_or_404(db, video_id)
    frame = db.get(Frame, frame_id)
    if frame is None or frame.video_id != video_id:
        raise AppError(404, "frame_not_found", "Frame not found.")
    path = get_settings().frames_dir / str(video_id) / frame.image_name  # names come from the DB
    if not path.is_file():
        raise AppError(404, "frame_not_found", "Frame image is no longer available.")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{video_id}/media")
def get_media(video_id: UUID, db: Session = Depends(get_db)) -> FileResponse:
    """Starlette's FileResponse serves Range requests, so the player can seek."""
    video = _get_or_404(db, video_id)
    path = get_settings().videos_dir / video.stored_name
    if not path.is_file():
        raise AppError(404, "media_not_found", "Video file is no longer available.")
    return FileResponse(path, media_type=video.mime_type or "video/mp4")


@router.post("/{video_id}/process", response_model=VideoStatusOut, status_code=202)
def process(video_id: UUID, db: Session = Depends(get_db)) -> Video:
    from app.workers.tasks import process_video_task

    video = _get_or_404(db, video_id)
    if video.status in (VideoStatus.queued.value, VideoStatus.processing.value):
        raise AppError(409, "already_processing", "Video is already queued or processing.")
    video.status = VideoStatus.queued.value
    video.progress = 0.0
    video.audio_status = video.visual_status = "pending"
    video.audio_error = video.visual_error = video.error_summary = None
    db.commit()
    try:
        process_video_task.delay(str(video.id))
    except Exception as exc:  # broker down
        video.status = VideoStatus.uploaded.value
        db.commit()
        log.warning("enqueue_failed", error=type(exc).__name__)
        raise AppError(503, "queue_unavailable", "Processing queue is unavailable.") from exc
    db.refresh(video)
    return video


@router.delete("/{video_id}", status_code=204)
def delete_video(video_id: UUID, db: Session = Depends(get_db)) -> None:
    video = _get_or_404(db, video_id)
    if video.status == VideoStatus.processing.value:
        raise AppError(409, "video_busy", "Video is being processed; try again when it finishes.")
    s = get_settings()
    try:
        index.delete_video(video.id)
    except Exception as exc:  # don't leave orphan vectors behind a "deleted" record
        log.warning("vector_delete_failed", error=type(exc).__name__)
        raise AppError(503, "index_unavailable", "Could not remove search data; try again.") from exc
    (s.videos_dir / video.stored_name).unlink(missing_ok=True)
    shutil.rmtree(s.frames_dir / str(video.id), ignore_errors=True)
    shutil.rmtree(s.transcripts_dir / str(video.id), ignore_errors=True)
    db.delete(video)  # FK cascade removes segments, chunks, frames, events
    db.commit()
    log.info("video_deleted", video_id=str(video_id))
