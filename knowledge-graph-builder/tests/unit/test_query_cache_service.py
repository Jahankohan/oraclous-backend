"""
Unit tests for QueryCacheService.

All tests use mocked Redis clients — no live Redis required.
pytest-asyncio auto mode is set in tests/unit/conftest.py.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

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
    def test_different_graph_id_produces_different_key(self):
        svc = QueryCacheService(redis_client=None)
        key_a = svc._cache_key("graph-aaa", "what is X?", "vector_cypher")
        key_b = svc._cache_key("graph-bbb", "what is X?", "vector_cypher")
        assert key_a != key_b

    def test_different_retriever_type_produces_different_key(self):
        svc = QueryCacheService(redis_client=None)
        key_a = svc._cache_key("graph-abc", "what is X?", "vector")
        key_b = svc._cache_key("graph-abc", "what is X?", "hybrid")
        assert key_a != key_b

    def test_same_inputs_produce_same_key(self):
        svc = QueryCacheService(redis_client=None)
        key_a = svc._cache_key("graph-xyz", "hello world", "vector_cypher")
        key_b = svc._cache_key("graph-xyz", "hello world", "vector_cypher")
        assert key_a == key_b

    def test_key_format_contains_graph_id(self):
        svc = QueryCacheService(redis_client=None)
        key = svc._cache_key("tenant-123", "query", "vector")
        assert key.startswith("qcache:tenant-123:")

    def test_query_normalisation_lower_strip(self):
        """Different case / leading-trailing whitespace → same key."""
        svc = QueryCacheService(redis_client=None)
        key_a = svc._cache_key("g1", "  What Is X?  ", "vector")
        key_b = svc._cache_key("g1", "what is x?", "vector")
        assert key_a == key_b

    def test_different_query_text_produces_different_key(self):
        svc = QueryCacheService(redis_client=None)
        key_a = svc._cache_key("g1", "question one", "vector")
        key_b = svc._cache_key("g1", "question two", "vector")
        assert key_a != key_b


# ---------------------------------------------------------------------------
# get() tests
# ---------------------------------------------------------------------------


class TestGet:
    async def test_cache_miss_returns_none(self):
        r = _mock_redis(get_return=None)
        svc = QueryCacheService(r)
        result = await svc.get("g1", "query", "vector")
        assert result is None

    async def test_cache_hit_returns_dict(self):
        payload = {"answer": "42", "confidence": 0.9}
        r = _mock_redis(get_return=payload)
        svc = QueryCacheService(r)
        result = await svc.get("g1", "query", "vector")
        assert result == payload

    async def test_connection_error_returns_none(self):
        import redis.exceptions as redis_exc

        r = _mock_redis(raise_on_get=redis_exc.ConnectionError("refused"))
        svc = QueryCacheService(r)
        result = await svc.get("g1", "query", "vector")
        assert result is None  # never raises

    async def test_timeout_error_returns_none(self):
        import redis.exceptions as redis_exc

        r = _mock_redis(raise_on_get=redis_exc.TimeoutError("timed out"))
        svc = QueryCacheService(r)
        result = await svc.get("g1", "query", "vector")
        assert result is None

    async def test_none_redis_client_returns_none(self):
        svc = QueryCacheService(redis_client=None)
        result = await svc.get("g1", "query", "vector")
        assert result is None


# ---------------------------------------------------------------------------
# set() tests
# ---------------------------------------------------------------------------


class TestSet:
    async def test_set_calls_redis_with_ttl(self):
        r = _mock_redis()
        svc = QueryCacheService(r)
        await svc.set("g1", "query", "vector", {"answer": "hi"}, ttl=60)
        r.set.assert_called_once()
        call_kwargs = r.set.call_args
        # third positional-or-keyword arg is ex=ttl
        assert call_kwargs.kwargs.get("ex") == 60 or call_kwargs.args[-1] == 60

    async def test_set_default_ttl_is_300(self):
        r = _mock_redis()
        svc = QueryCacheService(r)
        await svc.set("g1", "query", "vector", {"answer": "hi"})
        r.set.assert_called_once()
        call_kwargs = r.set.call_args
        assert call_kwargs.kwargs.get("ex") == 300

    async def test_connection_error_silently_ignored(self):
        import redis.exceptions as redis_exc

        r = _mock_redis(raise_on_set=redis_exc.ConnectionError("refused"))
        svc = QueryCacheService(r)
        # Must not raise
        await svc.set("g1", "query", "vector", {"answer": "hi"})

    async def test_timeout_error_silently_ignored(self):
        import redis.exceptions as redis_exc

        r = _mock_redis(raise_on_set=redis_exc.TimeoutError("timeout"))
        svc = QueryCacheService(r)
        await svc.set("g1", "query", "vector", {"answer": "hi"})

    async def test_none_redis_client_silently_ignored(self):
        svc = QueryCacheService(redis_client=None)
        await svc.set("g1", "query", "vector", {"answer": "hi"})


# ---------------------------------------------------------------------------
# Round-trip: set then get returns same result
# ---------------------------------------------------------------------------


class TestRoundTrip:
    async def test_second_call_returns_cached_result(self):
        """Simulate a two-call sequence via a simple in-memory store."""
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

        # First call: cache miss → nothing stored yet
        miss = await svc.get("g1", "hello", "vector")
        assert miss is None

        # Store it
        await svc.set("g1", "hello", "vector", payload)

        # Second call: cache hit
        hit = await svc.get("g1", "hello", "vector")
        assert hit == payload


# ---------------------------------------------------------------------------
# invalidate_graph() tests
# ---------------------------------------------------------------------------


class TestInvalidateGraph:
    async def test_deletes_matching_keys_not_others(self):
        """SCAN returns keys for graph-A; keys for graph-B must not be touched."""
        # Simulate SCAN: one batch with two keys for graph-A, then done (cursor=0).
        keys_for_a = [b"qcache:graph-A:abc", b"qcache:graph-A:def"]

        r = MagicMock()
        r.scan = AsyncMock(
            side_effect=[
                (0, keys_for_a),  # cursor=0 means iteration complete
            ]
        )
        r.delete = AsyncMock(return_value=2)

        svc = QueryCacheService(r)
        deleted = await svc.invalidate_graph("graph-A")

        assert deleted == 2
        r.delete.assert_called_once_with(*keys_for_a)

    async def test_invalidate_returns_zero_when_no_keys(self):
        r = _mock_redis()  # scan returns (0, []) by default
        svc = QueryCacheService(r)
        deleted = await svc.invalidate_graph("graph-B")
        assert deleted == 0

    async def test_invalidate_handles_multi_page_scan(self):
        """SCAN must iterate until cursor returns to 0."""
        batch1 = [b"qcache:g:key1", b"qcache:g:key2"]
        batch2 = [b"qcache:g:key3"]

        r = MagicMock()
        # First call returns cursor=99 (non-zero → continue)
        # Second call returns cursor=0 (done)
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

    async def test_connection_error_returns_zero_not_raises(self):
        import redis.exceptions as redis_exc

        r = MagicMock()
        r.scan = AsyncMock(side_effect=redis_exc.ConnectionError("refused"))

        svc = QueryCacheService(r)
        deleted = await svc.invalidate_graph("graph-err")
        assert deleted == 0

    async def test_none_redis_client_returns_zero(self):
        svc = QueryCacheService(redis_client=None)
        deleted = await svc.invalidate_graph("graph-X")
        assert deleted == 0

    async def test_scan_pattern_scoped_to_graph_id(self):
        """The SCAN match pattern must include graph_id to avoid cross-tenant reads."""
        r = MagicMock()
        r.scan = AsyncMock(return_value=(0, []))

        svc = QueryCacheService(r)
        await svc.invalidate_graph("my-graph")

        # Verify the match pattern used in the SCAN call
        call_kwargs = r.scan.call_args
        pattern = call_kwargs.kwargs.get("match") or call_kwargs.args[1]
        assert "my-graph" in pattern
        assert pattern.startswith("qcache:my-graph:")
