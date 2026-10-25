#  Video RAG

Ask a question about a video and jump to the exact moment that answers it.

Upload a video. The app transcribes the speech, reads on-screen text, indexes both (plus the
frames themselves), and answers questions with citations that point at a specific timestamp.
Clicking a citation seeks the player to that moment. It runs on a CPU-only machine with no
paid services required.

> **Status:** works locally end to end. It has not been deployed publicly yet.

## How it works

```
 upload ──▶ FastAPI ──▶ Celery worker (one video at a time)
                          │
          audio branch ───┤─── faster-whisper ──▶ transcript ──▶ time-aware chunks ─┐
                          │                                                         ├─▶ Qdrant
          visual branch ──┴─── sampled frames + scene changes ──▶ Tesseract OCR ────┤   (3 indexes)
                                                  └──────────────▶ CLIP embeddings ─┘
 question ──▶ embed ──▶ search 3 indexes ──▶ fuse (RRF) ──▶ LLM answers from evidence only
                                                           └─▶ backend rebuilds citations
```

- **Three indexes:** transcript text, on-screen (OCR) text and CLIP image vectors. Results
  are fused with weighted Reciprocal Rank Fusion, near-duplicate moments are collapsed, and
  transcript hits are narrowed to the best-matching sentence.
- **Grounded answers:** the model sees numbered evidence and may only cite `[E#]` labels.
  The backend rebuilds every citation (id, video, timestamps) from the retrieved records, so
  a timestamp the model invents never reaches the client. An answer with no valid citation
  is replaced by a refusal.
- **Independent branches:** audio and visual processing fail separately. If one fails the
  video is `partial` and stays searchable.
- **Swappable LLM:** local Ollama by default; Groq or any OpenAI-compatible endpoint by
  configuration. Only answer generation can go to the cloud. Transcription, OCR and
  embeddings always run locally.

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI, SQLAlchemy 2, Pydantic v2 |
| Queue | Celery + Redis |
| Storage | PostgreSQL (metadata), Qdrant (vectors), local disk (media, frames) |
| Speech | faster-whisper (CPU, int8) |
| Vision | PySceneDetect, OpenCV, Tesseract OCR, CLIP ViT-B/32 |
| Text embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| LLM | Ollama `llama3.2:3b` (default), or Groq / OpenAI-compatible |
| Frontend | React 18, Vite, GSAP |

## Requirements

- Python 3.11
- Node.js 18+
- Docker (for PostgreSQL, Redis, Qdrant)
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` on `PATH`
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (found automatically in the
  default Windows install location, otherwise put it on `PATH`)
- [Ollama](https://ollama.com/) with `ollama pull llama3.2:3b` (skip if you use a cloud LLM)
- About 16 GB RAM is comfortable. No GPU needed.

## Quick start

```bash
# 1. Infrastructure
cp .env.example .env                  # set POSTGRES_PASSWORD
docker compose up -d                  # PostgreSQL :5433, Redis :6379, Qdrant :6333

# 2. Backend
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell; Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env  # keep DATABASE_URL in sync with the root .env
cd backend
python -c "from app.db.session import get_engine; from app.db.models import Base; Base.metadata.create_all(get_engine())"
uvicorn app.main:app --port 8000

# 3. Worker (second terminal, from backend/). On Windows the solo pool is required.
celery -A app.workers.celery_app worker --pool=solo --concurrency=1 -l info

# 4. Frontend (third terminal)
cd frontend
npm install
npm run dev                           # http://localhost:5173  (proxies /api to :8000)
```

Open `http://localhost:5173/` for the landing page and `http://localhost:5173/#/app` for
the workspace. Check `GET /api/v1/diagnostics` if something is missing; it reports which
dependencies and models are reachable.

The first processed video downloads the Whisper and embedding models (a few hundred MB),
so it is much slower than later ones.

## Configuration

Settings live in `backend/.env`. `backend/.env.example` shows the main ones; the full list
with defaults is in `backend/app/core/config.py`.

| Variable | Purpose |
|---|---|
| `DEMO_ACCESS_TOKEN` | Required `X-Demo-Token` header on protected endpoints. Empty disables auth (local use only). |
| `UPLOADS_ENABLED` | Set `false` for a read-only public demo. |
| `LLM_PROVIDER` | `ollama` (default), `groq`, or `openai_compat` |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Cloud LLM settings. Keep the key out of the repo. |
| `WHISPER_MODEL` | `tiny` / `base` / `small`. Larger is slower and more accurate. |
| `FRAME_SAMPLE_INTERVAL_SECONDS` | Seconds between sampled frames (default 8). |
| `MAX_UPLOAD_MB`, `MAX_VIDEO_DURATION_SECONDS` | Upload limits (500 MB, 15 min by default). |
| `OCR_MAX_WIDTH`, `OCR_MIN_CONFIDENCE` | OCR resolution and word-confidence cutoff. |

To use Groq:

```
LLM_PROVIDER=groq
LLM_API_KEY=<your key>
```

## API

All routes are under `/api/v1`. Protected routes need `X-Demo-Token` when a token is set
(`?token=` is also accepted for `<video>` and `<img>` URLs).

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Dependency reachability (no auth) |
| GET | `/diagnostics` | Models, limits and warnings |
| POST | `/videos` | Upload a video (validated, size and duration limited) |
| GET | `/videos`, `/videos/{id}` | Library and metadata |
| GET | `/videos/{id}/status` | Overall and per-branch status, stage, progress |
| POST | `/videos/{id}/process` | Queue (or re-run) processing |
| GET | `/videos/{id}/transcript`, `/frames` | Timestamped segments, frames with OCR text |
| GET | `/videos/{id}/media` | Stream the video (range requests) |
| DELETE | `/videos/{id}` | Remove the video, files and vectors |
| POST | `/search` | Ranked evidence across transcript, OCR and frames |
| POST | `/ask` | Answer with structured citations, or a refusal |

## Development

```bash
cd backend
ruff check app tests
pytest
```

Unit tests cover chunking, citation validation, result fusion and de-duplication, sentence
targeting and OCR cleaning. They need no running services.

## Evaluation

Answer quality is evaluated with [RAGAS](https://docs.ragas.io) (faithfulness, answer
relevancy, context precision and context recall). See [`evals/`](evals/README.md) for how to
run it.

Measured on 28 questions over 4 videos (2026-10-25, Groq `qwen/qwen3.8-27b`):

| Metric | Result |
|---|---|
| Latency (`/ask`) p50 / p95 | 1.0 s / 1.4 s |
| Abstention accuracy (unanswerable questions flagged `insufficient_evidence`) | 2 / 6 (33%) |
| Abstention accuracy, judged by reading the answer text | 6 / 6 (100%) |
| Answerable questions wrongly refused | 0 / 22 |

## Security notes

- Upload type, size and duration are validated; stored files get server-generated names and
  no filesystem paths are returned to clients.
- Evidence text (transcripts, OCR) is treated as untrusted data in the LLM prompt.
- For any public deployment: set `DEMO_ACCESS_TOKEN`, turn uploads off, put the API behind
  a reverse proxy that sets forwarded-IP headers (the built-in rate limiter keys on the
  client IP), keep PostgreSQL, Redis and Qdrant off the public network, and cap your LLM
  spend.


