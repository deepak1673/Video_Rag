from fastapi import Header, Query, Request

from app.core.config import get_settings
from app.core.security import check_token, client_key, rate_limiter


def require_access(
    request: Request,
    x_demo_token: str | None = Header(default=None),
    token: str | None = Query(default=None),  # for <video src>/<img src> which cannot set headers
) -> None:
    """Token check + per-client rate limit for protected endpoints."""
    check_token(x_demo_token or token)
    rate_limiter.check(client_key(request), get_settings().rate_limit_per_minute)
