"""Multimodal retrieval: transcript text, OCR text and CLIP frame vectors, fused with
weighted Reciprocal Rank Fusion. Scores below the configured floors never enter fusion,
so an irrelevant query returns nothing instead of the "least bad" matches."""

import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import TranscriptChunk, TranscriptSegment
from app.db.session import session_scope
from app.retrieval import index
from app.retrieval.embeddings import embed_clip_texts, embed_texts

_NEARBY_SECONDS = 4.0  # frames closer than this (same video) are treated as one moment


@dataclass
class Evidence:
    evidence_id: str
    video_id: str
    source_type: str  # transcript | frame
    start: float
    end: float
    snippet: str
    modalities: list[str] = field(default_factory=list)  # transcript | ocr | visual
    best_score: float = 0.0  # strongest raw cosine among contributing hits
    fused_score: float = 0.0
    context: str = ""  # fuller text for the LLM; `snippet` is what a citation shows


def _rrf(rank: int, weight: float, k: int) -> float:
    return weight / (k + rank)


def search(query: str, video_ids: list[str] | None = None, top_k: int | None = None) -> list[Evidence]:
    s = get_settings()
    top_k = top_k or s.retrieval_top_k
    pool = top_k * 3  # over-fetch so dedupe still leaves top_k

    text_vec = embed_texts([query])[0]
    clip_vec = embed_clip_texts([query])[0]
    runs = [
        (index.TRANSCRIPT, "transcript", s.weight_transcript,
         index.search(index.TRANSCRIPT, text_vec, pool, video_ids, s.min_text_score)),
        (index.OCR, "ocr", s.weight_ocr,
         index.search(index.OCR, text_vec, pool, video_ids, s.min_text_score)),
        (index.VISUAL, "visual", s.weight_visual,
         index.search(index.VISUAL, clip_vec, pool, video_ids, s.min_visual_score)),
    ]

    merged: dict[tuple[str, str], Evidence] = {}
    for _collection, modality, weight, hits in runs:
        for rank, h in enumerate(hits, start=1):
            kind = "transcript" if modality == "transcript" else "frame"
            ev = merged.get((kind, h.evidence_id))
            if ev is None:
                p = h.payload
                ev = merged[(kind, h.evidence_id)] = Evidence(
                    evidence_id=h.evidence_id, video_id=h.video_id, source_type=kind,
                    start=p["start"], end=p["end"], snippet=p.get("text", ""),
                )
            ev.modalities.append(modality)
            ev.best_score = max(ev.best_score, h.score)
            ev.fused_score += _rrf(rank, weight, s.rrf_k)
            if modality == "ocr":
                ev.snippet = h.payload.get("text", ev.snippet)

    ranked = sorted(merged.values(), key=lambda e: e.fused_score, reverse=True)
    top = select_balanced(_dedupe(ranked), top_k)
    _focus_transcripts(top, text_vec, query)
    for ev in top:
        if ev.source_type == "frame":
            ev.snippet = focus_text(ev.snippet, query)
    return top


def select_balanced(deduped: list[Evidence], top_k: int) -> list[Evidence]:
    """Take the top_k results while reserving up to half the slots for transcript passages.

    On screen recordings the OCR text of a whole page repeats the query terms constantly
    and would otherwise crowd out the narration that actually explains them."""
    transcript = [e for e in deduped if e.source_type == "transcript"]
    frames = [e for e in deduped if e.source_type != "transcript"]
    n_t = min(len(transcript), max(top_k // 2, top_k - len(frames)))
    n_f = min(len(frames), top_k - n_t)
    chosen = transcript[:n_t] + frames[:n_f]
    return sorted(chosen, key=lambda e: e.fused_score, reverse=True)


def focus_text(text: str, query: str, max_words: int = 40) -> str:
    """The stretch of up to `max_words` words that shares the most content words with the
    query (earliest wins ties). Whole-screen OCR is mostly irrelevant for any one question."""
    words = text.split()
    if len(words) <= max_words:
        return text
    terms = content_terms(query)
    if not terms:
        return " ".join(words[:max_words])
    hits = [1 if (content_terms(w) & terms) else 0 for w in words]
    window = sum(hits[:max_words])
    best_i, best_hits = 0, window
    for i in range(1, len(words) - max_words + 1):
        window += hits[i + max_words - 1] - hits[i - 1]
        if window > best_hits:
            best_i, best_hits = i, window
    return " ".join(words[best_i:best_i + max_words])


_STOPWORDS = frozenset(
    "a an the is are was were be been do does did how what when where which who why whom "
    "of to in on at for from by with about into this that these those it its and or but "
    "you your we our they their i me my can could should would will there here any".split()
)
_WORD = re.compile(r"[a-z0-9]+")
WINDOW_SEGMENTS = 2  # a cited moment spans this many consecutive sentences
LEXICAL_WEIGHT = 0.5
FIRST_SENTENCE_WEIGHT = 0.2


def _stem(word: str) -> str:
    """Crude prefix stem so 'download' matches 'downloaded'/'downloading'."""
    return word[:5] if len(word) > 5 else word


def content_terms(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def best_window(texts: list[str], vectors: list[list[float]], query_vec: list[float],
                query: str, size: int = WINDOW_SEGMENTS) -> tuple[int, int]:
    """Index range [i, j) of the consecutive sentences that best answer the query.

    Score = best per-sentence cosine in the window + a bonus for the share of the query's
    content words the window contains, + a smaller bonus when the window's *first* sentence
    is itself relevant (so the citation starts where the answer starts, not one sentence
    early). Meaning alone is unreliable for short sentences; shared wording alone misses
    paraphrase. Remaining ties go to the earliest window."""
    terms = content_terms(query)
    sims = [sum(a * b for a, b in zip(v, query_vec, strict=True)) for v in vectors]

    def share(text: str) -> float:
        return len(terms & content_terms(text)) / len(terms) if terms else 0.0

    best, best_score = (0, min(size, len(texts))), float("-inf")
    for i in range(len(texts)):
        j = min(i + size, len(texts))
        score = (max(sims[i:j]) + LEXICAL_WEIGHT * share(" ".join(texts[i:j]))
                 + FIRST_SENTENCE_WEIGHT * share(texts[i]))
        if score > best_score + 1e-9:
            best, best_score = (i, j), score
    return best


def _focus_transcripts(evidence: list[Evidence], query_vec: list[float], query: str) -> None:
    """Narrow each transcript chunk (~30 s) to its best-matching sentences so a citation
    jumps to the right moment. The chunk text stays as `context` for the model."""
    chunks = [e for e in evidence if e.source_type == "transcript"]
    if not chunks:
        return
    with session_scope() as db:
        for ev in chunks:
            ev.context = ev.snippet
            chunk = db.get(TranscriptChunk, uuid.UUID(ev.evidence_id))
            if chunk is None or len(chunk.segment_idxs) < 2:
                continue
            segs = list(db.scalars(
                select(TranscriptSegment)
                .where(TranscriptSegment.video_id == chunk.video_id,
                       TranscriptSegment.idx.in_(chunk.segment_idxs))
                .order_by(TranscriptSegment.idx)
            ))
            if len(segs) < 2:
                continue
            texts = [s.text for s in segs]
            i, j = best_window(texts, embed_texts(texts), query_vec, query)
            ev.start, ev.end = segs[i].start_seconds, segs[j - 1].end_seconds
            ev.snippet = " ".join(texts[i:j])


SAME_SCREEN_JACCARD = 0.8


def _word_set(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def same_screen(a: str, b: str) -> bool:
    """True when two OCR texts are (almost) the same word set: a screen that stayed put."""
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= SAME_SCREEN_JACCARD


def _overlaps(a: Evidence, b: Evidence) -> bool:
    if a.video_id != b.video_id or a.source_type != b.source_type:
        return False
    if a.source_type == "frame":
        # A tutorial often sits on one screen for minutes; those frames are one moment.
        return abs(a.start - b.start) < _NEARBY_SECONDS or same_screen(a.snippet, b.snippet)
    return a.start < b.end and b.start < a.end


def _dedupe(ranked: list[Evidence]) -> list[Evidence]:
    kept: list[Evidence] = []
    for ev in ranked:
        if not any(_overlaps(ev, k) for k in kept):
            kept.append(ev)
    return kept
