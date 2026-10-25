from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class VideoOut(BaseModel):
    """Public view of a video. No stored file names or filesystem paths."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    original_filename: str
    mime_type: str
    size_bytes: int
    duration_seconds: float | None
    width: int | None
    height: int | None
    frame_rate: float | None
    has_audio: bool
    status: str
    audio_status: str
    visual_status: str
    current_stage: str | None
    progress: float
    error_summary: str | None
    is_demo: bool
    created_at: datetime
    processed_at: datetime | None


class VideoStatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    audio_status: str
    visual_status: str
    current_stage: str | None
    progress: float
    audio_error: str | None
    visual_error: str | None
    error_summary: str | None
