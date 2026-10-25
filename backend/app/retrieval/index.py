"""Qdrant access. Three cosine collections, one per evidence modality. Point ids are the
deterministic ids from Postgres, so upserts are idempotent and retries never duplicate."""

import uuid
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.retrieval.embeddings import CLIP_DIM, TEXT_DIM

log = get_logger("index")

TRANSCRIPT = "transcript_text"
OCR = "ocr_text"
VISUAL = "frames_visual"

_DIMS = {TRANSCRIPT: TEXT_DIM, OCR: TEXT_DIM, VISUAL: CLIP_DIM}
# OCR and visual points reuse the frame id: they sit in separate collections.


@dataclass(frozen=True)
class Hit:
    collection: str
    evidence_id: str  # chunk id (transcript) or frame id (ocr / visual)
    video_id: str
    score: float
    payload: dict


@lru_cache
def client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, timeout=20)


def ensure_collections() -> None:
    c = client()
    existing = {col.name for col in c.get_collections().collections}
    for name, dim in _DIMS.items():
        if name in existing:
            continue
        c.create_collection(name, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
        c.create_payload_index(name, "video_id", qm.PayloadSchemaType.KEYWORD)
        log.info("collection_created", name=name)


def upsert(collection: str, points: list[tuple[uuid.UUID, list[float], dict]]) -> None:
    if not points:
        return
    client().upsert(
        collection,
        points=[qm.PointStruct(id=str(pid), vector=vec, payload=pl) for pid, vec, pl in points],
        wait=True,  # never report ready before vectors are durable
    )


def delete_video(video_id: uuid.UUID, collections: tuple[str, ...] | None = None) -> None:
    flt = qm.Filter(must=[qm.FieldCondition(key="video_id", match=qm.MatchValue(value=str(video_id)))])
    for name in collections or tuple(_DIMS):
        client().delete(name, points_selector=qm.FilterSelector(filter=flt), wait=True)


def count_for_video(collection: str, video_id: uuid.UUID) -> int:
    flt = qm.Filter(must=[qm.FieldCondition(key="video_id", match=qm.MatchValue(value=str(video_id)))])
    return client().count(collection, count_filter=flt, exact=True).count


def search(collection: str, vector: list[float], top_k: int, video_ids: list[str] | None = None,
           min_score: float = 0.0) -> list[Hit]:
    flt = None
    if video_ids:
        flt = qm.Filter(must=[qm.FieldCondition(key="video_id", match=qm.MatchAny(any=video_ids))])
    try:
        res = client().query_points(collection, query=vector, limit=top_k, query_filter=flt,
                                    score_threshold=min_score or None, with_payload=True)
    except Exception as exc:
        log.warning("qdrant_search_failed", error=type(exc).__name__)
        raise AppError(503, "index_unavailable", "Search index is unavailable.") from exc
    return [
        Hit(collection, p.payload.get("evidence_id", str(p.id)), p.payload["video_id"],
            float(p.score), p.payload)
        for p in res.points
    ]
