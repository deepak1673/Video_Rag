"""Validated application settings.

Relative paths (DATA_DIR, .env) resolve from the repository root, never from the
current working directory of the shell.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_secret_key: str = "dev-insecure-secret"
    demo_access_token: str = ""
    cors_origins: str = "http://localhost:5173"
    uploads_enabled: bool = True

    database_url: str = "postgresql+psycopg://video_rag:change_me_locally@127.0.0.1:5433/video_rag"
    redis_url: str = "redis://127.0.0.1:6379/0"
    qdrant_url: str = "http://127.0.0.1:6333"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: float = 180.0

    # Answer-generation provider. Only generation can go to the cloud; transcription,
    # OCR and embeddings always stay local.
    llm_provider: Literal["ollama", "groq", "openai_compat"] = "ollama"
    llm_base_url: str = ""  # blank -> provider default
    llm_model: str = ""  # blank -> provider default
    llm_api_key: str = ""
    llm_timeout_seconds: float = 60.0

    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "sentence-transformers/clip-ViT-B-32"

    data_dir: Path = Path("data")
    max_upload_mb: int = Field(500, ge=1)
    max_video_duration_seconds: int = Field(900, ge=1)
    max_concurrent_video_jobs: int = Field(1, ge=1)
    frame_sample_interval_seconds: float = Field(8.0, gt=0)
    ocr_enabled: bool = True
    scene_detection_enabled: bool = True
    retrieval_top_k: int = Field(8, ge=1, le=50)
    min_free_disk_gb: float = 5.0
    rate_limit_per_minute: int = Field(60, ge=1)
    ffmpeg_timeout_seconds: int = 600
    ocr_timeout_seconds: int = 20
    ocr_max_width: int = Field(1600, ge=320)  # OCR runs on a temporary frame at up to this width
    ocr_min_confidence: float = Field(55.0, ge=0, le=100)

    # Retrieval / fusion tuning
    rrf_k: int = 60
    weight_transcript: float = 1.0
    weight_ocr: float = 0.8
    weight_visual: float = 0.6
    chunk_target_seconds: float = 30.0
    chunk_max_chars: int = 900
    min_text_score: float = 0.25  # cosine floor for text evidence to count as relevant
    # CLIP text-image cosine floor. Measured on this library: unrelated queries top out at
    # ~0.19-0.215 and real matches start ~0.28; the 0.22-0.24 band is mostly junk.
    min_visual_score: float = 0.25

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, v: Path) -> Path:
        return (v if v.is_absolute() else REPO_ROOT / v).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_is_local(self) -> bool:
        return self.llm_provider == "ollama"

    @property
    def llm_effective_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url.rstrip("/")
        if self.llm_provider == "groq":
            return "https://api.groq.com/openai/v1"
        return self.ollama_base_url.rstrip("/")

    @property
    def llm_effective_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        if self.llm_provider == "groq":
            return "llama-3.3-70b-versatile"
        return self.ollama_model

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def ensure_dirs(self) -> None:
        for d in (self.videos_dir, self.frames_dir, self.transcripts_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
