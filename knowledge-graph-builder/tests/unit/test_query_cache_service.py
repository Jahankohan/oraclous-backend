"""
Unit tests for QueryCacheService (TASK-028 / STORY-009).

All tests use mocked Redis clients — no live Redis required.
pytest-asyncio auto mode is configured in tests/unit/conftest.py.

Coverage:
- Cache key uniqueness: different graph_id → different key (cross-tenant isolation)
- Cache key uniqueness: different retriever_type → different key
- Cache hit: second call returns same result
- Cache miss: returns None
- invalidate_graph(): deletes all keys matching qcache:{graph_id}:*, not others
- Redis ConnectionError in get() → returns None (not exception)
- Redis ConnectionError in set() → silently ignored
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.query_cache_service import QueryCacheService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_redis(
    get_return=None,
    scan_side_effect=None,
    raise_on_get=None,
    raise_on_set=None,
) -> MagicMock:
    """Build a minimal async Redis mock."""
    r = MagicMock()
    if raise_on_get:
        r.get = AsyncMock(side_effect=raise_on_get)
    else:
        r.get = AsyncMock(
            return_value=json.dumps(get_return) if get_return is not None else None
        )
    if raise_on_set:
        r.set = AsyncMock(side_effect=raise_on_set)
    else:
        r.set = AsyncMock(return_value=True)
    r.delete = AsyncMock(return_value=1)
    if scan_side_effect is not None:
        r.scan = AsyncMock(side_effect=scan_side_effect)
    else:
        # Default: empty scan (no keys)
        r.scan = AsyncMock(return_value=(0, []))
    return r


# ---------------------------------------------------------------------------
# Cache key uniqueness tests
# ---------------------------------------------------------------------------


class TestCacheKeyUniqueness:
    @pytest.mark.unit
    def test_different_graph_id_produces_different_key(self):
        """Cross-tenant isolation: different graph_id must never share a cache key."""
        svc = QueryCacheService(redis_client=None)
        key_a = svc._cache_key("graph-aaa", "what is X?", "vector_cypher")
        key_b = svc._cache_key("graph-bbb", "what is X?", "vector_cypher")
        assert key_a != key_b

    @pytest.mark.unit
    def test_different_retriever_type_produces_different_key(self):
        """Different retriever_type values must produce different cache keys."""
        svc = QueryCacheService(redis_client=None)
        key_a = svc._cache_key("graph-abc", "what is X?", "vector")
        key_b = svc._cache_key("graph-abc", "what is X?", "hybrid")
        assert key_a != key_b

    @pytest.mark.unit
    def test_same_inputs_produce_same_key(self):
        """Identical inputs must always resolve to the same deterministic key."""
        svc = QueryCacheService(redis_client=None)
        key_a = svc._cache_key("graph-xyz", "hello world", "vector_cypher")
        key_b = svc._cache_key("graph-xyz", "hello world", "vector_cypher")
        assert key_a == key_b

    @pytest.mark.unit
    def test_key_format_contains_graph_id_prefix(self):
        """Key must start with qcache:{graph_id}: to guarantee tenant scope."""
        svc = QueryCacheService(redis_client=None)
        key = svc._cache_key("tenant-123", "query", "vector")
        assert key.startswith("qcache:tenant-123:")

    @pytest.mark.unit
    def test_query_normalisation_lower_strip(self):
        """Case and leading/trailing whitespace must not produce different keys."""
        svc = QueryCacheService(redis_client=None)
        key_a = svc._cache_key("g1", "  What Is X?  ", "vector")
        key_b = svc._cache_key("g1", "what is x?", "vector")
        assert key_a == key_b

    @pytest.mark.unit
    def test_different_query_text_produces_different_key(self):
        """Different query texts must produce distinct keys."""
        svc = QueryCacheService(redis_client=None)
        key_a = svc._cache_key("g1", "question one", "vector")
        key_b = svc._cache_key("g1", "question two", "vector")
        assert key_a != key_b

    @pytest.mark.unit
    def test_different_graph_id_different_key_binary_cross_check(self):
        """Explicit cross-tenant check: graph-A key must not equal graph-B key."""
        svc = QueryCacheService(redis_client=None)
        same_query = "who founded the company?"
        same_retriever = "vector_cypher"
        key_a = svc._cache_key("tenant-A", same_query, same_retriever)
        key_b = svc._cache_key("tenant-B", same_query, same_retriever)
        assert key_a != key_b
        # Also verify the prefix isolation
        assert "tenant-A" in key_a
        assert "tenant-B" in key_b


# ---------------------------------------------------------------------------
# get() tests
# ---------------------------------------------------------------------------


class TestGet:
    @pytest.mark.unit
    async def test_cache_miss_returns_none(self):
        """Redis returns None → get() must return None."""
        r = _mock_redis(get_return=None)
        svc = QueryCacheService(r)
        result = await svc.get("g1", "query", "vector")
        assert result is None

    @pytest.mark.unit
    async def test_cache_hit_returns_dict(self):
        """Redis returns serialised payload → get() must return the deserialized dict."""
        payload = {"answer": "42", "confidence": 0.9}
        r = _mock_redis(get_return=payload)
        svc = QueryCacheService(r)
        result = await svc.get("g1", "query", "vector")
        assert result == payload

    @pytest.mark.unit
    async def test_connection_error_returns_none_not_exception(self):
        """Redis ConnectionError in get() must return None, never raise."""
        import redis.exceptions as redis_exc

        r = _mock_redis(raise_on_get=redis_exc.ConnectionError("refused"))
        svc = QueryCacheService(r)
        result = await svc.get("g1", "query", "vector")
        assert result is None  # must NOT raise

    @pytest.mark.unit
    async def test_timeout_error_returns_none_not_exception(self):
        """Redis TimeoutError in get() must return None, never raise."""
        import redis.exceptions as redis_exc

        r = _mock_redis(raise_on_get=redis_exc.TimeoutError("timed out"))
        svc = QueryCacheService(r)
        result = await svc.get("g1", "query", "vector")
        assert result is None

    @pytest.mark.unit
    async def test_none_redis_client_returns_none(self):
        """When redis_client is None (disabled mode), get() must return None immediately."""
        svc = QueryCacheService(redis_client=None)
        result = await svc.get("g1", "query", "vector")
        assert result is None


# ---------------------------------------------------------------------------
# set() tests
# ---------------------------------------------------------------------------


class TestSet:
    @pytest.mark.unit
    async def test_set_calls_redis_with_custom_ttl(self):
        """set() must pass the TTL argument through to Redis."""
        r = _mock_redis()
        svc = QueryCacheService(r)
        await svc.set("g1", "query", "vector", {"answer": "hi"}, ttl=60)
        r.set.assert_called_once()
        call_kwargs = r.set.call_args
        assert call_kwargs.kwargs.get("ex") == 60

    @pytest.mark.unit
    async def test_set_default_ttl_is_300(self):
        """When no TTL is passed, set() must default to 300 seconds."""
        r = _mock_redis()
        svc = QueryCacheService(r)
        await svc.set("g1", "query", "vector", {"answer": "hi"})
        r.set.assert_called_once()
        call_kwargs = r.set.call_args
        assert call_kwargs.kwargs.get("ex") == 300

    @pytest.mark.unit
    async def test_connection_error_silently_ignored(self):
        """Redis ConnectionError in set() must be swallowed — the request path must not block."""
        import redis.exceptions as redis_exc

        r = _mock_redis(raise_on_set=redis_exc.ConnectionError("refused"))
        svc = QueryCacheService(r)
        # Must not raise
        await svc.set("g1", "query", "vector", {"answer": "hi"})

    @pytest.mark.unit
    async def test_timeout_error_silently_ignored(self):
        """Redis TimeoutError in set() must be swallowed silently."""
        import redis.exceptions as redis_exc

        r = _mock_redis(raise_on_set=redis_exc.TimeoutError("timeout"))
        svc = QueryCacheService(r)
        await svc.set("g1", "query", "vector", {"answer": "hi"})

    @pytest.mark.unit
    async def test_none_redis_client_silently_ignored(self):
        """When redis_client is None, set() must be a no-op."""
        svc = QueryCacheService(redis_client=None)
        await svc.set("g1", "query", "vector", {"answer": "hi"})


# ---------------------------------------------------------------------------
# Round-trip: set then get returns same result (cache hit on second call)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.unit
    async def test_second_call_returns_cached_result(self):
        """
        Cache hit: after set(), a second get() with the same parameters must
        return the stored result (not None).
        """
        store: dict[str, str] = {}

        async def fake_get(key):
            return store.get(key)

        async def fake_set(key, value, ex=None):
            store[key] = value

        r = MagicMock()
        r.get = AsyncMock(side_effect=fake_get)
        r.set = AsyncMock(side_effect=fake_set)

        svc = QueryCacheService(r)
        payload = {"answer": "cached answer", "confidence": 0.85}

        # First call: cache miss
        miss = await svc.get("g1", "hello", "vector")
        assert miss is None

        # Store the result
        await svc.set("g1", "hello", "vector", payload)

        # Second call: cache hit
        hit = await svc.get("g1", "hello", "vector")
        assert hit == payload

    @pytest.mark.unit
    async def test_cross_tenant_no_cache_bleed(self):
        """
        Tenant isolation: a cache entry for graph-A must not appear when querying
        the same text for graph-B.
        """
        store: dict[str, str] = {}

        async def fake_get(key):
            return store.get(key)

        async def fake_set(key, value, ex=None):
            store[key] = value

        r = MagicMock()
        r.get = AsyncMock(side_effect=fake_get)
        r.set = AsyncMock(side_effect=fake_set)

        svc = QueryCacheService(r)
        payload_a = {"answer": "answer for tenant A"}

        await svc.set("tenant-A", "shared query text", "vector", payload_a)

        # tenant-B must NOT get tenant-A's cached result
        result_b = await svc.get("tenant-B", "shared query text", "vector")
        assert result_b is None

        # tenant-A still sees its own result
        result_a = await svc.get("tenant-A", "shared query text", "vector")
        assert result_a == payload_a


# ---------------------------------------------------------------------------
# invalidate_graph() tests
# ---------------------------------------------------------------------------


class TestInvalidateGraph:
    @pytest.mark.unit
    async def test_deletes_matching_keys_not_others(self):
        """
        invalidate_graph() must delete qcache:{graph_id}:* keys and nothing else.
        Other tenant keys must remain untouched.
        """
        keys_for_a = [b"qcache:graph-A:abc", b"qcache:graph-A:def"]

        r = MagicMock()
        r.scan = AsyncMock(return_value=(0, keys_for_a))
        r.delete = AsyncMock(return_value=2)

        svc = QueryCacheService(r)
        deleted = await svc.invalidate_graph("graph-A")

        assert deleted == 2
        r.delete.assert_called_once_with(*keys_for_a)

    @pytest.mark.unit
    async def test_invalidate_returns_zero_when_no_keys(self):
        """When no keys match, invalidate_graph() must return 0."""
        r = _mock_redis()  # scan returns (0, []) by default
        svc = QueryCacheService(r)
        deleted = await svc.invalidate_graph("graph-B")
        assert deleted == 0

    @pytest.mark.unit
    async def test_invalidate_handles_multi_page_scan(self):
        """
        SCAN must iterate until cursor returns to 0 — multi-page scans must
        accumulate all deleted key counts correctly.
        """
        batch1 = [b"qcache:g:key1", b"qcache:g:key2"]
        batch2 = [b"qcache:g:key3"]

        r = MagicMock()
        # cursor=99 → continue; cursor=0 → done
        r.scan = AsyncMock(
            side_effect=[
                (99, batch1),
                (0, batch2),
            ]
        )
        r.delete = AsyncMock(return_value=1)

        svc = QueryCacheService(r)
        deleted = await svc.invalidate_graph("g")

        assert deleted == 3  # 2 + 1
        assert r.delete.call_count == 2

    @pytest.mark.unit
    async def test_connection_error_returns_zero_not_raises(self):
        """Redis ConnectionError during invalidation must return 0, never raise."""
        import redis.exceptions as redis_exc

        r = MagicMock()
        r.scan = AsyncMock(side_effect=redis_exc.ConnectionError("refused"))

        svc = QueryCacheService(r)
        deleted = await svc.invalidate_graph("graph-err")
        assert deleted == 0

    @pytest.mark.unit
    async def test_none_redis_client_returns_zero(self):
        """When redis_client is None, invalidate_graph() must return 0 immediately."""
        svc = QueryCacheService(redis_client=None)
        deleted = await svc.invalidate_graph("graph-X")
        assert deleted == 0

    @pytest.mark.unit
    async def test_scan_pattern_scoped_to_graph_id(self):
        """The SCAN match pattern must include graph_id to prevent cross-tenant reads."""
        r = MagicMock()
        r.scan = AsyncMock(return_value=(0, []))

        svc = QueryCacheService(r)
        await svc.invalidate_graph("my-graph")

        call_kwargs = r.scan.call_args
        pattern = call_kwargs.kwargs.get("match") or (
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        assert pattern is not None, "scan must be called with a match pattern"
        assert "my-graph" in pattern
        assert pattern.startswith("qcache:my-graph:")

    @pytest.mark.unit
    async def test_invalidate_graph_a_does_not_touch_graph_b_keys(self):
        """
        invalidate_graph('graph-A') must issue a SCAN with a pattern that
        cannot match graph-B keys — verified by checking the pattern argument.
        """
        r = MagicMock()
        r.scan = AsyncMock(return_value=(0, []))

        svc = QueryCacheService(r)
        await svc.invalidate_graph("graph-A")

        call_kwargs = r.scan.call_args
        pattern = call_kwargs.kwargs.get("match") or (
            call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        )
        # Pattern must be scoped to graph-A, not a wildcard for all tenants
        assert "graph-A" in pattern
        assert "graph-B" not in pattern
        assert pattern != "qcache:*"  # must NOT be a global wildcard
