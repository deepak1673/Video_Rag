"""Relational schema. Video timestamps are float SECONDS on the original media timeline;
database timestamps are timezone-aware UTC."""

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class VideoStatus(str, enum.Enum):
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    ready = "ready"
    partial = "partial"
    failed = "failed"
    deleting = "deleting"


class BranchStatus(str, enum.Enum):
    pending = "pending"
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    retrying = "retrying"
    failed = "failed"
    skipped = "skipped"


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_name: Mapped[str] = mapped_column(String(80))  # file name inside data/videos
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    has_audio: Mapped[bool] = mapped_column(default=True)

    status: Mapped[str] = mapped_column(String(20), default=VideoStatus.uploaded.value, index=True)
    audio_status: Mapped[str] = mapped_column(String(20), default=BranchStatus.pending.value)
    visual_status: Mapped[str] = mapped_column(String(20), default=BranchStatus.pending.value)
    current_stage: Mapped[str | None] = mapped_column(String(60), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1
    audio_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    visual_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_demo: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    segments: Mapped[list["TranscriptSegment"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", passive_deletes=True
    )
    chunks: Mapped[list["TranscriptChunk"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", passive_deletes=True
    )
    frames: Mapped[list["Frame"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", passive_deletes=True
    )
    events: Mapped[list["ProcessingEvent"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", passive_deletes=True
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    idx: Mapped[int] = mapped_column(Integer)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    speaker_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    video: Mapped[Video] = relationship(back_populates="segments")

    __table_args__ = (Index("ix_segments_video_start", "video_id", "start_seconds"),)


class TranscriptChunk(Base):
    """Time-aware group of segments. `id` is deterministic (uuid5) and is also the
    Qdrant point id and the public evidence id, so re-indexing is idempotent."""

    __tablename__ = "transcript_chunks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    video_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    idx: Mapped[int] = mapped_column(Integer)
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    segment_idxs: Mapped[list] = mapped_column(JSON, default=list)

    video: Mapped[Video] = relationship(back_populates="chunks")


class Frame(Base):
    """Sampled frame. `id` is deterministic (uuid5); also the evidence id for both
    the visual point and the OCR point (the point ids differ, see indexing)."""

    __tablename__ = "frames"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    video_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    idx: Mapped[int] = mapped_column(Integer)
    timestamp_seconds: Mapped[float] = mapped_column(Float)
    image_name: Mapped[str] = mapped_column(String(80))  # file inside data/frames/<video_id>
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    scene_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    video: Mapped[Video] = relationship(back_populates="frames")

    __table_args__ = (Index("ix_frames_video_ts", "video_id", "timestamp_seconds"),)


class ProcessingEvent(Base):
    __tablename__ = "processing_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    branch: Mapped[str] = mapped_column(String(20))  # audio | visual | system
    stage: Mapped[str] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(20))
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    safe_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    video: Mapped[Video] = relationship(back_populates="events")
