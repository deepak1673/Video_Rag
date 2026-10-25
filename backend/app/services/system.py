"""Dependency reachability checks shared by /health and /diagnostics."""

import shutil
import subprocess
from pathlib import Path

import httpx
from sqlalchemy import text

from app.core.config import get_settings


def find_tesseract() -> str | None:
    """Tesseract is often installed but not on PATH on Windows."""
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def check_database() -> bool:
    try:
        from app.db.session import get_engine

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def check_redis() -> bool:
    try:
        import redis

        return bool(redis.Redis.from_url(get_settings().redis_url, socket_timeout=2).ping())
    except Exception:
        return False


def check_qdrant() -> bool:
    try:
        r = httpx.get(f"{get_settings().qdrant_url}/readyz", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def ollama_models() -> list[str] | None:
    """Names of locally pulled models, or None if Ollama is unreachable."""
    try:
        r = httpx.get(f"{get_settings().ollama_base_url}/api/tags", timeout=2.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def ollama_status() -> dict:
    models = ollama_models()
    configured = get_settings().ollama_model
    if models is None:
        return {"reachable": False, "model_available": False}
    base = configured.split(":")[0]
    available = configured in models or (":" not in configured and any(
        m.split(":")[0] == base for m in models
    ))
    return {"reachable": True, "model_available": available}


def llm_status() -> dict:
    """Active generation provider. Cloud providers are not pinged (saves quota)."""
    s = get_settings()
    if s.llm_is_local:
        return {"provider": "ollama", **ollama_status()}
    return {"provider": s.llm_provider, "configured": bool(s.llm_api_key)}


def check_binary(name: str) -> bool:
    if name == "tesseract":
        return find_tesseract() is not None
    return shutil.which(name) is not None


def binary_version(name: str) -> str | None:
    exe = find_tesseract() if name == "tesseract" else shutil.which(name)
    if not exe:
        return None
    flag = "-version" if name in ("ffmpeg", "ffprobe") else "--version"
    try:
        out = subprocess.run(
            [exe, flag], capture_output=True, text=True, timeout=5, check=False
        )
        return (out.stdout or out.stderr).splitlines()[0][:80]
    except Exception:
        return None


def disk_free_gb() -> float:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return round(shutil.disk_usage(settings.data_dir).free / 1024**3, 1)


def public_health() -> dict:
    """Booleans only: safe to expose without authentication."""
    s = get_settings()
    llm = llm_status()
    deps = {
        "database": check_database(),
        "redis": check_redis(),
        "qdrant": check_qdrant(),
        "llm": llm["reachable"] if s.llm_is_local else llm["configured"],
        "ffmpeg": check_binary("ffmpeg"),
        "tesseract": check_binary("tesseract"),
    }
    core_ok = all(deps[k] for k in ("database", "redis", "qdrant"))
    return {
        "status": "ok" if core_ok else "degraded",
        "dependencies": deps,
        "inference": "local-cpu" if s.llm_is_local else "cloud-llm",
    }
