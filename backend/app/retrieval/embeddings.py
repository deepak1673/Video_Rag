"""Local embedding models, loaded lazily once per process. Vectors are L2-normalised so
Qdrant cosine scores are directly comparable to the configured score floors."""

import threading
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger("embeddings")

TEXT_DIM = 384  # all-MiniLM-L6-v2
CLIP_DIM = 512  # clip-ViT-B-32

_models: dict[str, object] = {}
_lock = threading.Lock()


def _load(name: str):
    with _lock:
        if name not in _models:
            from sentence_transformers import SentenceTransformer

            log.info("embedding_model_loading", model=name)
            _models[name] = SentenceTransformer(name, device="cpu")
        return _models[name]


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []
    model = _load(get_settings().text_embedding_model)
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False)
    return vecs.tolist()


def embed_clip_texts(texts: list[str]) -> list[list[float]]:
    """Query text into the CLIP image-text space."""
    if not texts:
        return []
    model = _load(get_settings().clip_model)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def embed_images(paths: list[Path], batch_size: int = 16) -> list[list[float]]:
    if not paths:
        return []
    from PIL import Image

    model = _load(get_settings().clip_model)
    out: list[list[float]] = []
    for i in range(0, len(paths), batch_size):
        imgs = []
        for p in paths[i : i + batch_size]:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())
        out.extend(model.encode(imgs, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=False).tolist())
    return out
