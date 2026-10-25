"""Demo-token protection and a small in-memory per-client rate limiter."""

import hmac
import time
from collections import defaultdict, deque

from fastapi import Request

from app.core.config import get_settings
from app.core.errors import AppError


def check_token(provided: str | None) -> None:
    expected = get_settings().demo_access_token
    if not expected:
        return  # local-only mode: token disabled
    if not provided:
        raise AppError(401, "auth_required", "Missing demo access token.")
    if not hmac.compare_digest(provided, expected):
        raise AppError(403, "auth_invalid", "Invalid demo access token.")


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int, window: float = 60.0) -> None:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            raise AppError(429, "rate_limited", "Too many requests. Please slow down.")
        q.append(now)

    def reset(self) -> None:
        self._hits.clear()


rate_limiter = RateLimiter()


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"
