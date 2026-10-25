from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    video_ids: list[UUID] | None = Field(default=None, max_length=20)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    video_ids: list[UUID] | None = Field(default=None, max_length=20)
    top_k: int | None = Field(default=None, ge=1, le=20)


class EvidenceOut(BaseModel):
    evidence_id: str
    video_id: str
    source_type: str  # transcript | frame
    modalities: list[str]
    start_seconds: float
    end_seconds: float
    snippet: str
    score: float
    thumbnail_url: str | None = None


class SearchResponse(BaseModel):
    results: list[EvidenceOut]


class CitationOut(BaseModel):
    label: int
    evidence_id: str
    video_id: str
    source_type: str
    modalities: list[str]
    start_seconds: float
    end_seconds: float
    snippet: str
    thumbnail_url: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    insufficient_evidence: bool
    elapsed_ms: int


class SegmentOut(BaseModel):
    idx: int
    start_seconds: float
    end_seconds: float
    text: str


class FrameOut(BaseModel):
    id: UUID
    idx: int
    timestamp_seconds: float
    ocr_text: str | None
    scene_id: int | None
    thumbnail_url: str
