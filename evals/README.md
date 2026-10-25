# Evaluation (RAGAS)

Scores the `/ask` pipeline end to end with [RAGAS](https://docs.ragas.io). The runner calls the
running API over HTTP, so it lives in its own environment and does not touch the backend's
pinned dependencies.

## Metrics

| Metric | Question it answers |
|---|---|
| `faithfulness` | Is every claim in the answer supported by the retrieved evidence? |
| `answer_relevancy` | Does the answer actually address the question? |
| `llm_context_precision_with_reference` | Are the useful evidence items ranked above the noise? |
| `context_recall` | Did retrieval surface what the reference answer needs? |
| `abstention_accuracy` | Of the unanswerable questions, how many did the app refuse? (not RAGAS) |
| `false_abstentions` | Answerable questions the app wrongly refused. (not RAGAS) |

Plus p50/p95 latency of `/search` + `/ask`. Refused answers are excluded from the RAGAS
scores (a refusal has nothing to judge) and counted in `false_abstentions` instead.

## Setup

```powershell
python -m venv .venv-eval
.venv-eval\Scripts\pip install -r evals/requirements.txt
```

## Judge LLM

The judge is configured separately from the app's answer LLM so a small local model does not
grade itself. RAGAS prompts need a capable model; a 3B model produces many unparseable
judgements. Defaults to Groq `openai/gpt-oss-120b`.

```powershell
$env:EVAL_LLM_API_KEY  = "<groq key>"
# optional:
$env:EVAL_LLM_BASE_URL = "https://api.groq.com/openai/v1"   # or http://127.0.0.1:11434/v1 for Ollama
$env:EVAL_LLM_MODEL    = "openai/gpt-oss-120b"
```

Note that the question, answer and retrieved snippets (transcript/OCR text) are sent to the
judge provider. Use a local judge if the videos are private.

## Run

1. Start the stack (API, worker, Qdrant, ...) with your videos already ingested.
2. Copy `dataset.example.jsonl` to `dataset.jsonl` and fill it in (see below).
3. Run:

```powershell
.venv-eval\Scripts\python evals\run_eval.py evals\dataset.jsonl
# retrieval + answers only, no judge LLM needed:
.venv-eval\Scripts\python evals\run_eval.py evals\dataset.jsonl --collect-only
```

Results land in `evals/results/<timestamp>.json` (summary, raw answers, per-question scores).
Set `DEMO_ACCESS_TOKEN` if the API has a demo token, `EVAL_API_URL` for a non-default host.

## Dataset format

One JSON object per line:

| Field | Notes |
|---|---|
| `question` | required |
| `reference` | required when answerable: a short, manually verified answer |
| `answerable` | `false` for questions the videos do not answer; default `true` |
| `video_ids` | optional list of video UUIDs to scope the question |
| `tags` | optional, free-form (`speech`, `slide`, `visual`, ...) |

Include unanswerable questions: without them abstention accuracy is meaningless. Write
references by watching the video, not by reading the model's output.
