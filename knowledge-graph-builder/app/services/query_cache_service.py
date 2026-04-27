"""
Query Cache Service — Redis-backed cache for chat query results.

Key design decisions:
- Key format: qcache:{graph_id}:{sha256_hash}
  - graph_id in every key guarantees cross-tenant isolation (Architecture Rule: graph_id on every key)
  - sha256 of normalised (lower-stripped) query + retriever_type prevents collisions
- SCAN for invalidation — never FLUSHDB or KEYS (production-safe)
- All Redis errors are swallowed — cache is advisory; never blocks the request path
- async Redis client (redis.asyncio) for FastAPI; sync wrapper for Celery callers
"""

import hashlib
import json
import logging

try:
    import redis.asyncio as aioredis
    import redis.exceptions as redis_exc
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]
    redis_exc = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class QueryCacheService:
    """Redis-backed cache for chat query results.

    Usage (FastAPI path — async):
        cache = QueryCacheService(redis_client)
        cached = await cache.get(graph_id, query_text, retriever_type)
        if cached:
            return ChatResponse(**cached, cache_hit=True)
        ...generate response...
        await cache.set(graph_id, query_text, retriever_type, response.dict())

    Usage (Celery path — after ingest completes):
        deleted = await cache.invalidate_graph(graph_id)
    """

    def __init__(self, redis_client):
        """Initialise with an async Redis client.

        Args:
            redis_client: A redis.asyncio.Redis (or compatible) instance.
                          Pass None to operate in a no-op / cache-disabled mode —
                          all operations become silent no-ops.
        """
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _cache_key(self, graph_id: str, query_text: str, retriever_type: str) -> str:
        """Build a deterministic, tenant-scoped cache key.

        The key encodes (graph_id, normalised query, retriever_type) so that:
        - Different tenants never share a cache entry (graph_id prefix)
        - Whitespace/case variations in queries hit the same key
        - Different retriever types produce different cached results

        Format: qcache:{graph_id}:{sha256hex}
        """
        normalised = query_text.lower().strip()
        payload = f"{graph_id}|{normalised}|{retriever_type}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"qcache:{graph_id}:{digest}"

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def get(
        self, graph_id: str, query_text: str, retriever_type: str
    ) -> dict | None:
        """Return a cached result dict, or None on miss / Redis unavailable.

        Never raises — any Redis error returns None so the caller falls through
        to live query execution.
        """
        if self._redis is None:
            return None
        key = self._cache_key(graph_id, query_text, retriever_type)
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except (redis_exc.ConnectionError, redis_exc.TimeoutError) as exc:
            logger.warning("QueryCache.get: Redis unavailable (%s) — cache miss", exc)
            return None
        except Exception as exc:
            logger.warning("QueryCache.get: unexpected error (%s) — cache miss", exc)
            return None

    async def set(
        self,
        graph_id: str,
        query_text: str,
        retriever_type: str,
        result: dict,
        ttl: int = 300,
    ) -> None:
        """Cache result for TTL seconds (default: 5 minutes).

        Silently ignores any Redis error — the request path must not be blocked
        by a cache write failure.
        """
        if self._redis is None:
            return
        key = self._cache_key(graph_id, query_text, retriever_type)
        try:
            serialised = json.dumps(result, default=str)
            await self._redis.set(key, serialised, ex=ttl)
            logger.debug(
                "QueryCache.set: cached key=%s ttl=%ds graph_id=%s",
                key,
                ttl,
                graph_id,
            )
        except (redis_exc.ConnectionError, redis_exc.TimeoutError) as exc:
            logger.warning(
                "QueryCache.set: Redis unavailable (%s) — result not cached", exc
            )
        except Exception as exc:
            logger.warning(
                "QueryCache.set: unexpected error (%s) — result not cached", exc
            )

    async def invalidate_graph(self, graph_id: str) -> int:
        """Delete all cached query results for graph_id.

        Uses SCAN with a pattern to find keys — never FLUSHDB or KEYS.
        FLUSHDB would wipe all tenants; KEYS blocks the Redis event loop on large keyspaces.

        Returns:
            Number of keys deleted (0 if Redis is down or no keys matched).
        """
        if self._redis is None:
            return 0
        pattern = f"qcache:{graph_id}:*"
        deleted = 0
        try:
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match=pattern, count=100
                )
                if keys:
                    await self._redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info(
                "QueryCache.invalidate_graph: deleted %d key(s) for graph_id=%s",
                deleted,
                graph_id,
            )
        except (redis_exc.ConnectionError, redis_exc.TimeoutError) as exc:
            logger.warning(
                "QueryCache.invalidate_graph: Redis unavailable (%s) — invalidation skipped",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "QueryCache.invalidate_graph: unexpected error (%s) — invalidation skipped",
                exc,
            )
        return deleted
