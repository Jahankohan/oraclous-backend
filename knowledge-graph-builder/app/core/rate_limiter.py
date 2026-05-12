"""
Rate limiting for the Knowledge Graph Builder API.

Two layers:
  1. ``limiter`` — flat per-endpoint IP-keyed limiter (slowapi) used on public
     endpoints that don't carry a tenant identity.
  2. ``TokenBucketRateLimiter`` — per-tenant token bucket backed by Redis,
     used on authenticated endpoints where a ``tenant_id`` is available.

Token bucket algorithm
----------------------
Tokens are refilled continuously at ``rate`` tokens/second up to ``burst``.
Each request consumes one token.  The bucket state is stored in Redis as a
hash with two fields — ``tokens`` (float) and ``last_refill`` (float epoch).
A Lua script executes the refill + consume atomically so the operation is
safe across multiple FastAPI workers.

Rate categories (defaults)
--------------------------
  read   — GET endpoints          60 req/min  (rate=1.0/s,   burst=20)
  write  — POST/PUT/DELETE        30 req/min  (rate=0.5/s,   burst=10)
  admin  — graph create/delete    10 req/min  (rate=0.167/s, burst=3)

Per-tenant overrides
--------------------
Optional ``rate_limit_config`` property on the tenant's ``:ServiceAccount``
or ``:Tenant`` Neo4j node.  Expected format::

    {
      "read":  {"rate": 2.0, "burst": 40},
      "write": {"rate": 1.0, "burst": 20},
      "admin": {"rate": 0.5, "burst": 5}
    }

The config is cached in Redis at key ``ratelimit_config:{tenant_id}`` for 60 s
so that Neo4j is not queried on every request.
"""

import json
import time
from typing import Any

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Flat slowapi limiter (non-tenant paths)
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Default token bucket parameters per endpoint category
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, dict[str, float]] = {
    "read": {"rate": 1.0, "burst": 20.0},  # 60/min, burst 20
    "write": {"rate": 0.5, "burst": 10.0},  # 30/min, burst 10
    "admin": {"rate": 0.167, "burst": 3.0},  # ~10/min, burst 3
}

_CONFIG_CACHE_TTL = 60  # seconds to cache per-tenant config in Redis
_BUCKET_TTL = 3600  # seconds before an idle bucket expires in Redis

# ---------------------------------------------------------------------------
# Atomic Lua script
# ---------------------------------------------------------------------------

_LUA_TOKEN_BUCKET = """
local key         = KEYS[1]
local rate        = tonumber(ARGV[1])
local burst       = tonumber(ARGV[2])
local now         = tonumber(ARGV[3])

local bucket      = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens      = tonumber(bucket[1]) or burst
local last_refill = tonumber(bucket[2]) or now

local elapsed = now - last_refill
tokens = math.min(burst, tokens + elapsed * rate)

if tokens >= 1 then
    tokens = tokens - 1
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return {1, math.floor(tokens)}
else
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    local retry_after = math.ceil((1 - tokens) / rate)
    return {0, retry_after}
end
"""


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter
# ---------------------------------------------------------------------------


class TokenBucketRateLimiter:
    """
    Per-tenant token bucket rate limiter backed by Redis.

    Parameters
    ----------
    redis_client:
        An async Redis client (e.g. ``redis.asyncio`` from the ``redis``
        package).  Must support ``eval``, ``get``, ``set``, and ``delete``.
    tenant_id:
        Opaque string that identifies the tenant.  Used as part of the Redis
        key so that each tenant has its own independent bucket.
    endpoint_type:
        One of ``"read"``, ``"write"``, or ``"admin"``.  Selects the default
        rate/burst values and the per-tenant override bucket to read.
    neo4j_driver:
        Optional async Neo4j driver used to look up per-tenant config.  When
        ``None`` only default limits are used.
    """

    def __init__(
        self,
        redis_client: Any,
        tenant_id: str,
        endpoint_type: str,
        neo4j_driver: Any = None,
    ) -> None:
        if endpoint_type not in _DEFAULTS:
            raise ValueError(
                f"endpoint_type must be one of {list(_DEFAULTS)}; got {endpoint_type!r}"
            )
        self._redis = redis_client
        self._tenant_id = tenant_id
        self._endpoint_type = endpoint_type
        self._neo4j_driver = neo4j_driver
        self._bucket_key = f"ratelimit:{tenant_id}:{endpoint_type}"
        self._config_key = f"ratelimit_config:{tenant_id}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def is_allowed(self) -> tuple[bool, dict[str, Any]]:
        """
        Consume one token from the bucket.

        Returns
        -------
        (allowed, headers)
            ``allowed`` is ``True`` when the request is within the rate limit.
            ``headers`` is a dict of HTTP headers to include in the response:

            On success::

                {
                    "X-RateLimit-Limit": "30",
                    "X-RateLimit-Remaining": "9",
                    "X-RateLimit-Reset": "1714214400",
                }

            On limit exceeded::

                {
                    "Retry-After": "12",
                    "X-RateLimit-Limit": "30",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "1714214412",
                }
        """
        rate, burst = await self._get_rate_and_burst()
        now = time.time()

        try:
            result = await self._redis.eval(
                _LUA_TOKEN_BUCKET,
                1,
                self._bucket_key,
                str(rate),
                str(burst),
                str(now),
            )
            allowed = bool(int(result[0]))
            second_value = int(result[1])
        except Exception as exc:
            # Redis unavailable — fail open with a warning so that a Redis
            # outage doesn't immediately block all API traffic.
            logger.warning(
                "Redis unavailable for rate limiting (tenant=%s): %s — failing open",
                self._tenant_id,
                exc,
            )
            return True, {}

        # Approximate reset epoch based on burst refill time.
        reset_epoch = int(now + burst / rate)
        limit_str = str(int(burst))

        if allowed:
            remaining = second_value
            headers = {
                "X-RateLimit-Limit": limit_str,
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_epoch),
            }
            return True, headers
        else:
            retry_after = second_value
            headers = {
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": limit_str,
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(now) + retry_after),
            }
            return False, headers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_rate_and_burst(self) -> tuple[float, float]:
        """
        Return (rate, burst) for this tenant and endpoint type.

        Lookup order:
          1. Redis config cache (TTL 60 s)
          2. Neo4j `:ServiceAccount` / `:Tenant` node ``rate_limit_config``
          3. Hard-coded defaults
        """
        defaults = _DEFAULTS[self._endpoint_type]

        try:
            cached = await self._redis.get(self._config_key)
        except Exception as exc:
            logger.warning(
                "Redis config cache read error (tenant=%s): %s", self._tenant_id, exc
            )
            return defaults["rate"], defaults["burst"]

        if cached:
            try:
                config = json.loads(cached)
                ep_cfg = config.get(self._endpoint_type, {})
                rate = float(ep_cfg.get("rate", defaults["rate"]))
                burst = float(ep_cfg.get("burst", defaults["burst"]))
                return rate, burst
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Malformed rate limit config cache (tenant=%s): %s",
                    self._tenant_id,
                    exc,
                )

        # Cache miss — try Neo4j.
        if self._neo4j_driver is not None:
            config = await self._load_config_from_neo4j()
            if config:
                try:
                    await self._redis.set(
                        self._config_key, json.dumps(config), ex=_CONFIG_CACHE_TTL
                    )
                except Exception as exc:
                    logger.warning(
                        "Redis config cache write error (tenant=%s): %s",
                        self._tenant_id,
                        exc,
                    )
                ep_cfg = config.get(self._endpoint_type, {})
                rate = float(ep_cfg.get("rate", defaults["rate"]))
                burst = float(ep_cfg.get("burst", defaults["burst"]))
                return rate, burst

        return defaults["rate"], defaults["burst"]

    async def _load_config_from_neo4j(self) -> dict[str, Any] | None:
        """
        Query Neo4j for the tenant's ``rate_limit_config`` property.

        Uses parameterised Cypher — never interpolates ``tenant_id``.
        Returns the parsed dict or ``None`` when not found.
        """
        query = """
        MATCH (n)
        WHERE (n:ServiceAccount OR n:Tenant)
          AND (n.tenant_id = $tenant_id OR n.id = $tenant_id)
          AND n.rate_limit_config IS NOT NULL
        RETURN n.rate_limit_config AS cfg
        LIMIT 1
        """
        try:
            async with self._neo4j_driver.session() as session:
                result = await session.run(query, tenant_id=self._tenant_id)
                record = await result.single()
                if record is None:
                    return None
                raw = record["cfg"]
                if isinstance(raw, str):
                    return json.loads(raw)
                if isinstance(raw, dict):
                    return raw
                return None
        except Exception as exc:
            logger.warning(
                "Neo4j rate limit config lookup failed (tenant=%s): %s",
                self._tenant_id,
                exc,
            )
            return None
