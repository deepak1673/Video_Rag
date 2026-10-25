"""Safe upload handling: size-capped streaming, server-generated file names."""

import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings
from app.core.errors import AppError
from app.ingestion.media import ALLOWED_EXTENSIONS

_CHUNK = 1024 * 1024


def safe_extension(filename: str | None) -> str:
    """Extension from the client name, checked against an allow-list. The client name is
    never used as a path: stored files get a generated name."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise AppError(415, "unsupported_type", f"Unsupported file type. Allowed: {allowed}.")
    return ext


async def save_upload(file: UploadFile) -> tuple[Path, str, int]:
    """Stream to data/videos, enforcing the size cap. Returns (path, stored_name, size)."""
    s = get_settings()
    s.ensure_dirs()
    ext = safe_extension(file.filename)
    stored_name = f"{uuid.uuid4().hex}{ext}"
    dest = s.videos_dir / stored_name
    size = 0
    try:
        with dest.open("wb") as fh:
            while chunk := await file.read(_CHUNK):
                size += len(chunk)
                if size > s.max_upload_bytes:
                    raise AppError(
                        413, "upload_too_large", f"Upload exceeds the {s.max_upload_mb} MB limit."
                    )
                fh.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise AppError(400, "empty_upload", "Uploaded file is empty.")
    return dest, stored_name, size
