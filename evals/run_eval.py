"""RAGAS evaluation for the Video RAG answer pipeline.

Talks to the running API over HTTP, so it needs none of the backend's dependencies:

    .venv-eval\\Scripts\\python evals\\run_eval.py evals\\dataset.jsonl

For every question it calls /search (the evidence the model is shown) and /ask (the answer),
then scores the answerable questions with RAGAS. Unanswerable questions are not scored by
RAGAS; they feed an abstention-accuracy number instead, because "refused correctly" is not
something faithfulness or relevancy can express.

The judge LLM is configured separately from the app's answer LLM (EVAL_* variables) so a
small local model is not grading its own homework.
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import httpx

# Mirrors backend/app/generation/answer.py: the LLM sees at most this many non-empty
# snippets, so those are the "retrieved contexts" RAGAS should judge against.
MAX_LLM_EVIDENCE = 6


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("question"):
            sys.exit(f"{path}:{n}: missing 'question'")
        if row.get("answerable", True) and not row.get("reference"):
            sys.exit(f"{path}:{n}: answerable questions need a 'reference' answer")
        rows.append(row)
    if not rows:
        sys.exit(f"{path} has no rows")
    return rows


def post_with_backoff(client: httpx.Client, path: str, payload: dict, tries: int = 6) -> httpx.Response:
    """POST, waiting out 429s (free-tier LLM limits surface through /ask as 429)."""
    for attempt in range(tries):
        r = client.post(path, json=payload)
        if r.status_code != 429 or attempt == tries - 1:
            r.raise_for_status()
            return r
        wait = float(r.headers.get("retry-after") or 15 * 2**attempt)
        print(f"  rate limited, retrying in {wait:.0f}s", flush=True)
        time.sleep(min(wait, 120))
    raise AssertionError("unreachable")


def collect(client: httpx.Client, row: dict, top_k: int | None) -> dict:
    body = {"video_ids": row["video_ids"]} if row.get("video_ids") else {}
    if top_k:
        body["top_k"] = top_k
    found = post_with_backoff(client, "/search", {"query": row["question"], **body})
    asked = post_with_backoff(client, "/ask", {"question": row["question"], **body})
    ask = asked.json()
    contexts = [r["snippet"] for r in found.json()["results"] if r["snippet"].strip()]
    return {
        **row,
        "response": ask["answer"],
        "abstained": ask["insufficient_evidence"],
        "contexts": contexts[:MAX_LLM_EVIDENCE],
        "citations": len(ask["citations"]),
        "latency_ms": int(asked.elapsed.total_seconds() * 1000),  # the /ask call alone
    }


def build_judge():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    base_url = os.environ.get("EVAL_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    model = os.environ.get("EVAL_LLM_MODEL", "openai/gpt-oss-120b")
    api_key = os.environ.get("EVAL_LLM_API_KEY") or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        sys.exit("Set EVAL_LLM_API_KEY (any non-empty value for a local Ollama /v1 endpoint).")
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0,
                     timeout=120, max_retries=3)
    # Same embedding model the app uses; answer_relevancy only needs similarity.
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # bypass_n: answer_relevancy asks for several generations per call (n>1), which Groq
    # and some other OpenAI-compatible providers reject; this makes RAGAS issue separate calls.
    return LangchainLLMWrapper(llm, bypass_n=True), LangchainEmbeddingsWrapper(emb), model


def score(samples: list[dict]) -> tuple[list[dict], str]:
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    llm, emb, judge_model = build_judge()
    dataset = EvaluationDataset.from_list([
        {"user_input": s["question"], "response": s["response"],
         "retrieved_contexts": s["contexts"], "reference": s["reference"]}
        for s in samples
    ])
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
                 LLMContextRecall()],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(timeout=180, max_workers=2),  # gentle on free-tier rate limits
        raise_exceptions=False,
    )
    return result.to_pandas().to_dict("records"), judge_model


def mean(values: list[float]) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float)) and v == v]  # drop NaN/None
    return round(statistics.fmean(clean), 3) if clean else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--api", default=os.environ.get("EVAL_API_URL", "http://127.0.0.1:8000/api/v1"))
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("evals/results"))
    ap.add_argument("--collect-only", action="store_true", help="skip RAGAS (no judge LLM needed)")
    ap.add_argument("--rescore", type=Path, metavar="RESULTS_JSON",
                    help="re-judge the answers saved in an earlier results file (no API calls)")
    args = ap.parse_args()

    if args.rescore:
        samples = json.loads(args.rescore.read_text(encoding="utf-8"))["samples"]
    else:
        rows = load_dataset(args.dataset)
        headers = ({"X-Demo-Token": os.environ["DEMO_ACCESS_TOKEN"]}
                   if os.environ.get("DEMO_ACCESS_TOKEN") else {})
        samples = []
        with httpx.Client(base_url=args.api, headers=headers, timeout=300) as client:
            for i, row in enumerate(rows, start=1):
                print(f"[{i}/{len(rows)}] {row['question']}", flush=True)
                samples.append(collect(client, row, args.top_k))

    answerable = [s for s in samples if s.get("answerable", True)]
    unanswerable = [s for s in samples if not s.get("answerable", True)]
    latencies = sorted(s["latency_ms"] for s in samples)

    summary: dict = {
        "date": date.today().isoformat(),
        "dataset": str(args.dataset),
        "questions": len(samples),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "latency_ms_p50": latencies[len(latencies) // 2],
        "latency_ms_p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        # answerable questions the app wrongly refused
        "false_abstentions": sum(s["abstained"] for s in answerable),
        "abstention_accuracy": (
            round(sum(s["abstained"] for s in unanswerable) / len(unanswerable), 3)
            if unanswerable else None
        ),
    }

    scored: list[dict] = []
    if not args.collect_only and answerable:
        # An abstention has no grounded answer to judge; scoring the refusal text would
        # just drag every metric down. Those are already counted in false_abstentions.
        to_score = [s for s in answerable if not s["abstained"]]
        if to_score:
            scored, judge = score(to_score)
            summary["judge_model"] = judge
            summary["scored"] = len(to_score)
            for name in ("faithfulness", "answer_relevancy", "llm_context_precision_with_reference",
                         "context_recall"):
                summary[name] = mean([r.get(name) for r in scored])

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (args.out / f"{stamp}.json").write_text(
        json.dumps({"summary": summary, "samples": samples, "scores": scored},
                   indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out / f'{stamp}.json'}")


if __name__ == "__main__":
    main()
