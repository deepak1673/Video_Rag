import time

from fastapi import APIRouter, Depends

from app.api.deps import require_access
from app.generation.answer import answer_question
from app.retrieval.search import search
from app.schemas.search import (
    AskRequest,
    AskResponse,
    CitationOut,
    EvidenceOut,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(tags=["search"], dependencies=[Depends(require_access)])


def thumbnail_url(source_type: str, video_id: str, evidence_id: str) -> str | None:
    if source_type != "frame":
        return None
    return f"/api/v1/videos/{video_id}/frames/{evidence_id}/thumbnail"


@router.post("/search", response_model=SearchResponse)
def search_endpoint(body: SearchRequest) -> SearchResponse:
    ids = [str(v) for v in body.video_ids] if body.video_ids else None
    results = search(body.query, ids, body.top_k)
    return SearchResponse(results=[
        EvidenceOut(
            evidence_id=e.evidence_id, video_id=e.video_id, source_type=e.source_type,
            modalities=e.modalities, start_seconds=e.start, end_seconds=e.end,
            snippet=e.snippet, score=round(e.best_score, 4),
            thumbnail_url=thumbnail_url(e.source_type, e.video_id, e.evidence_id),
        )
        for e in results
    ])


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest) -> AskResponse:
    started = time.perf_counter()
    ids = [str(v) for v in body.video_ids] if body.video_ids else None
    result = answer_question(body.question, ids, body.top_k)
    return AskResponse(
        answer=result.answer,
        citations=[
            CitationOut(
                label=c.label, evidence_id=c.evidence_id, video_id=c.video_id, source_type=c.source_type,
                modalities=c.modalities, start_seconds=c.start_seconds,
                end_seconds=c.end_seconds, snippet=c.snippet,
                thumbnail_url=thumbnail_url(c.source_type, c.video_id, c.evidence_id),
            )
            for c in result.citations
        ],
        insufficient_evidence=result.insufficient_evidence,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
