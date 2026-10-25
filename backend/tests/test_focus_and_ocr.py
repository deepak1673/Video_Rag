from app.ingestion.ocr import clean_words
from app.retrieval.search import (
    Evidence,
    _dedupe,
    best_window,
    content_terms,
    focus_text,
    same_screen,
    select_balanced,
)


def ev(eid, kind, fused):
    return Evidence(eid, "v", kind, 0.0, 1.0, "text", [kind], 0.5, fused)


def test_select_balanced_reserves_transcript_slots_against_ocr_flood():
    frames = [ev(f"f{i}", "frame", 1.0 - i * 0.01) for i in range(8)]
    transcript = [ev(f"t{i}", "transcript", 0.1 - i * 0.01) for i in range(4)]
    chosen = select_balanced(frames + transcript, 8)
    assert sum(e.source_type == "transcript" for e in chosen) == 4
    assert len(chosen) == 8


def test_select_balanced_uses_all_slots_when_one_kind_is_missing():
    frames = [ev(f"f{i}", "frame", 1.0 - i * 0.01) for i in range(10)]
    assert len(select_balanced(frames, 8)) == 8
    assert len(select_balanced([ev("t0", "transcript", 0.5)], 8)) == 1


def test_focus_text_picks_the_window_with_query_words():
    words = ["noise"] * 60 + ["the", "infobox", "template", "is", "here"] + ["noise"] * 60
    out = focus_text(" ".join(words), "What is an infobox template?", max_words=10)
    assert "infobox" in out and "template" in out and len(out.split()) == 10


def test_same_screen_detects_near_identical_ocr_but_not_different_screens():
    a = "Donate to Wikipedia Wikimedia Shop Content that violates any copyrights will be deleted"
    b = "Donate to Wikipedia Wikimedia Shop Content that violates any copyrights will be deleted today"
    c = "Product Launch Europe Q4 2026"
    assert same_screen(a, b)
    assert not same_screen(a, c)
    assert not same_screen("", a)


def test_dedupe_collapses_frames_showing_the_same_screen_far_apart_in_time():
    text = "Wikipedia sandbox page draft review Donate Wikimedia Shop Content copyrights"
    ranked = []
    for i, t in enumerate((164.0, 180.0, 188.0, 228.0)):
        e = ev(f"f{i}", "frame", 1.0 - i * 0.1)
        e.start = e.end = t
        e.snippet = text
        ranked.append(e)
    assert [e.evidence_id for e in _dedupe(ranked)] == ["f0"]


def test_focus_text_leaves_short_text_alone():
    assert focus_text("short ocr text", "anything") == "short ocr text"


def unit(*xs):
    n = sum(x * x for x in xs) ** 0.5
    return [x / n for x in xs]


def test_content_terms_drop_question_words_and_stem():
    assert content_terms("How do you download OpenRefine?") == {"downl", "openr"}
    assert "downl" in content_terms("You downloaded it")


def test_best_window_prefers_shared_wording_when_embeddings_tie():
    texts = ["Welcome to the tutorial.", "You can download OpenRefine for free.", "Then open Chrome."]
    same = unit(1, 0)
    i, j = best_window(texts, [same, same, same], same, "How do you download OpenRefine?")
    assert texts[i].startswith("You can download")


def test_best_window_uses_meaning_when_no_words_overlap():
    texts = ["A cat sat.", "Stars are distant suns.", "Rain fell."]
    vecs = [unit(1, 0), unit(0, 1), unit(1, 0)]
    i, j = best_window(texts, vecs, unit(0, 1), "astronomy facts")
    assert i <= 1 < j  # the chosen window contains the astronomy sentence


def test_best_window_handles_single_sentence():
    assert best_window(["Only one."], [unit(1, 0)], unit(1, 0), "one") == (0, 1)


def test_clean_words_drops_low_confidence_and_junk():
    words = ["File", "Edit", "xQ", "a", "", "Tools", "#"]
    conf = [96.0, 91.0, 20.0, 99.0, 95.0, 88.0, 90.0]
    assert clean_words(words, conf, 55.0) == "File Edit Tools"
