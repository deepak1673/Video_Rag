from fastapi import APIRouter

from app.services.system import public_health

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    """Public, booleans only: no paths, credentials or error details."""
    return public_health()
