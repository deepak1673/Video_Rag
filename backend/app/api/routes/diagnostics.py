from fastapi import APIRouter, Depends

from app.api.deps import require_access
from app.core.config import get_settings
from app.services import system

router = APIRouter(tags=["diagnostics"], dependencies=[Depends(require_access)])


@router.get("/diagnostics")
def diagnostics() -> dict:
    """Protected: model/service readiness and configured limits. No secrets or paths."""
    s = get_settings()
    llm = system.llm_status()
    free = system.disk_free_gb()
    warnings = []
    if free < s.min_free_disk_gb:
        warnings.append(f"Low free disk space ({free} GB < {s.min_free_disk_gb} GB).")
    if not system.check_binary("tesseract"):
        warnings.append("Tesseract not found: OCR will be skipped.")
    if s.llm_is_local:
        if not llm["reachable"]:
            warnings.append("Ollama is not reachable: question answering is unavailable.")
        elif not llm["model_available"]:
            warnings.append(
                f"Ollama model '{s.llm_effective_model}' is not pulled "
                f"(ollama pull {s.llm_effective_model})."
            )
    elif not llm["configured"]:
        warnings.append("LLM_API_KEY is not set: question answering is unavailable.")
    return {
        "mode": "cpu-only",
        "inference": "local" if s.llm_is_local else "cloud-llm",
        "services": {
            "database": system.check_database(),
            "redis": system.check_redis(),
            "qdrant": system.check_qdrant(),
            "llm": llm,
        },
        "tools": {
            "ffmpeg": system.binary_version("ffmpeg"),
            "ffprobe": system.binary_version("ffprobe"),
            "tesseract": system.binary_version("tesseract"),
        },
        "models": {
            "whisper": {"name": s.whisper_model, "device": s.whisper_device,
                        "compute_type": s.whisper_compute_type},
            "text_embedding": s.text_embedding_model,
            "clip": s.clip_model,
            "llm": s.llm_effective_model,
        },
        "limits": {
            "max_upload_mb": s.max_upload_mb,
            "max_video_duration_seconds": s.max_video_duration_seconds,
            "max_concurrent_video_jobs": s.max_concurrent_video_jobs,
            "frame_sample_interval_seconds": s.frame_sample_interval_seconds,
            "retrieval_top_k": s.retrieval_top_k,
            "uploads_enabled": s.uploads_enabled,
        },
        "disk_free_gb": free,
        "warnings": warnings,
    }
