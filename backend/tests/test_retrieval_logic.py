import uuid
from types import SimpleNamespace

from app.generation.answer import build_citations
from app.ingestion.chunking import build_chunks
from app.retrieval.search import Evidence, _dedupe


def seg(idx, start, end, text):
    return SimpleNamespace(idx=idx, start_seconds=start, end_seconds=end, text=text)


def ev(eid, kind, start, end, video="v1"):
    return Evidence(eid, video, kind, start, end, "text", [kind], 0.5, 0.1)


def test_chunks_split_on_target_duration_and_keep_times():
    segs = [seg(i, i * 10.0, i * 10.0 + 9.0, f"sentence {i}") for i in range(6)]
    chunks = build_chunks(uuid.uuid4(), segs)
    assert len(chunks) > 1
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 59.0
    assert [i for c in chunks for i in c.segment_idxs] == list(range(6))


def test_chunk_ids_are_deterministic():
    vid = uuid.uuid4()
    segs = [seg(0, 0.0, 5.0, "hello")]
    assert build_chunks(vid, segs)[0].id == build_chunks(vid, segs)[0].id


def test_citations_ignore_labels_the_model_invented():
    evidence = [ev("a", "transcript", 0, 5), ev("b", "frame", 6, 6)]
    cites = build_citations("Yes [E2]. Also [E9] and [E0].", evidence)
    assert [c.evidence_id for c in cites] == ["b"]


def test_uncited_answer_yields_no_citations():
    assert build_citations("The launch is in Q4.", [ev("a", "transcript", 0, 5)]) == []


def test_dedupe_collapses_nearby_frames_but_not_other_videos():
    ranked = [
        ev("f1", "frame", 10.0, 10.0),
        ev("f2", "frame", 12.0, 12.0),
        ev("f3", "frame", 11.0, 11.0, video="v2"),
        ev("t1", "transcript", 8.0, 12.0),
    ]
    kept = [e.evidence_id for e in _dedupe(ranked)]
    assert kept == ["f1", "f3", "t1"]
