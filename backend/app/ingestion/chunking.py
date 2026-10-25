"""Time-aware transcript chunking. Chunks keep the start/end of their segments and the
segment indexes they were built from; ids are deterministic so re-indexing is idempotent."""

import uuid
from dataclasses import dataclass

from app.core.config import get_settings

_CHUNK_NS = uuid.UUID("a3c4f0de-2f6b-4b39-8c1e-0b7d52c1d9aa")


@dataclass(frozen=True)
class Chunk:
    id: uuid.UUID
    idx: int
    start: float
    end: float
    text: str
    segment_idxs: list[int]


def chunk_id(video_id: uuid.UUID, idx: int) -> uuid.UUID:
    return uuid.uuid5(_CHUNK_NS, f"{video_id}:{idx}")


def build_chunks(video_id: uuid.UUID, segments: list) -> list[Chunk]:
    """`segments`: TranscriptSegment rows ordered by time."""
    s = get_settings()
    chunks: list[Chunk] = []
    cur: list = []

    def flush() -> None:
        if not cur:
            return
        i = len(chunks)
        chunks.append(Chunk(
            id=chunk_id(video_id, i), idx=i, start=cur[0].start_seconds,
            end=cur[-1].end_seconds, text=" ".join(x.text for x in cur),
            segment_idxs=[x.idx for x in cur],
        ))
        cur.clear()

    for seg in segments:
        if cur:
            span = seg.end_seconds - cur[0].start_seconds
            chars = sum(len(x.text) + 1 for x in cur) + len(seg.text)
            if span > s.chunk_target_seconds or chars > s.chunk_max_chars:
                flush()
        cur.append(seg)
    flush()
    return chunks
