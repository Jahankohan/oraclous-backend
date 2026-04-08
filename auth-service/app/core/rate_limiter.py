"""Rate limiting for the auth-service.

Two controls are applied to POST /service-token:
  1. IP-based: 10 requests/minute per client IP  (via slowapi)
  2. Prefix-based: 10 requests/minute per key_prefix extracted from the request body
     (via async Redis counter, implemented as a FastAPI dependency)

If Redis is unreachable the prefix-based check fails open (allows the request)
so that a Redis outage does not lock all service-account authentications out.
The IP-based slowapi limit uses its own internal storage (also Redis) and is
configured independently.
"""

import json
import logging

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- IP-based limiter (slowapi) -------------------------------------------
# storage_uri wires slowapi's internal counters to Redis so limits survive
# restarts and are shared across multiple uvicorn workers.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.REDIS_URL,
)

# --- Per-key_prefix limiter constants ----------------------------------------
_KEY_PREFIX_LIMIT = 10
_KEY_PREFIX_WINDOW_SECONDS = 60


async def enforce_key_prefix_rate_limit(request: Request) -> None:
    """FastAPI dependency: block >10 req/min for the same key_prefix.

    key_prefix is the first 12 characters of the ``api_key`` field in the
    request body (e.g. ``osk_AbCdEfGh1234``).  The body bytes are cached by
    Starlette so the endpoint's Pydantic parsing is not affected.
    """
    body_bytes = await request.body()
    try:
        data = json.loads(body_bytes)
        api_key: str = data.get("api_key", "") or ""
    except Exception:
        api_key = ""

    key_prefix = api_key[:12]
    if not key_prefix:
        # No prefix to rate-limit on; invalid-key handling is the endpoint's job
        return

    redis_client: aioredis.Redis = getattr(request.app.state, "redis", None)
    if redis_client is None:
        logger.warning("rate_limiter: Redis not available, skipping prefix check")
        return

    redis_key = f"rl:pfx:{key_prefix}"
    try:
        count = await redis_client.incr(redis_key)
        if count == 1:
            # First request in this window — set the expiry
            await redis_client.expire(redis_key, _KEY_PREFIX_WINDOW_SECONDS)
        if count > _KEY_PREFIX_LIMIT:
            ttl = await redis_client.ttl(redis_key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for this key prefix. Try again later.",
                headers={"Retry-After": str(max(ttl, 1))},
            )
    except HTTPException:
        raise
    except Exception as exc:
        # Redis error → fail open; log and allow the request through
        logger.error("rate_limiter: Redis error during prefix check: %s", exc)


# --- Custom 429 handler ---------------------------------------------------
# The default slowapi handler leaks the limit configuration in the response
# body (e.g. "Rate limit exceeded: 10 per 1 minute").  This handler returns a
# generic message so no rate-limit configuration is exposed to clients.

async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a generic 429 — never expose limit configuration in the body."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests"},
    )
