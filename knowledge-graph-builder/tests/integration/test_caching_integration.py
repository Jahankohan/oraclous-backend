"""
Integration tests for TASK-031 / STORY-009: Redis query cache.

DO NOT RUN NOW — these tests require all STORY-009 PRs to be merged to develop
and a running Docker environment (Neo4j + Redis + the KGB service).

Run after TASK-028, TASK-029, TASK-030 are all merged:
    docker compose exec knowledge-graph-builder python -m pytest \
        tests/integration/test_caching_integration.py -v -m integration

Coverage:
- First query: cache_hit: False, response time ~normal
- Second identical query: cache_hit: True, response time <1s (assert <1.0 second)
- Ingest new document: cache invalidated for that graph
- Third query: cache_hit: False (fresh result after invalidation)
- Cross-tenant: graph_id A's cache not visible from graph_id B
- Unique graph_id per test: f"test-task031-{uuid4().hex[:8]}"
- Cleanup in pytest teardown — never leave test graphs in Neo4j
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8003/api/v1"
TEST_PREFIX = "test-task031"


def _unique_graph_id() -> str:
    """Generate a unique test graph_id that will be cleaned up after the test."""
    return f"{TEST_PREFIX}-{uuid4().hex[:8]}"


@pytest.fixture
def graph_id():
    """Yield a unique graph_id for this test; cleaned up after the test completes."""
    import httpx

    gid = _unique_graph_id()
    yield gid

    # Teardown: delete the test graph from Neo4j to avoid leftover state
    try:
        httpx.delete(f"{BASE_URL}/graphs/{gid}", timeout=10)
    except Exception:
        pass  # Best-effort cleanup — do not fail tests on teardown errors


@pytest.fixture
def auth_headers():
    """Return auth headers for test requests.

    Update this fixture when the integration environment uses a real auth token.
    """
    return {"X-API-Key": "test-integration-key"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _post_chat(graph_id: str, query: str, headers: dict) -> tuple[dict, float]:
    """POST to /chat and return (response_body, elapsed_seconds)."""
    import httpx

    payload = {
        "graph_id": graph_id,
        "query": query,
        "retriever_type": "vector_cypher",
    }
    start = time.perf_counter()
    response = httpx.post(
        f"{BASE_URL}/chat",
        json=payload,
        headers=headers,
        timeout=30,
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return response.json(), elapsed


def _post_ingest(graph_id: str, content: str, headers: dict) -> dict:
    """POST to /graphs/{id}/ingest and return response body."""
    import httpx

    payload = {
        "graph_id": graph_id,
        "content": content,
        "content_type": "text/plain",
    }
    response = httpx.post(
        f"{BASE_URL}/graphs/{graph_id}/ingest",
        json=payload,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# 1. First query: cache_hit: False
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.docker
def test_first_query_cache_miss(graph_id, auth_headers):
    """
    The first query for a given (graph_id, query, retriever_type) combination
    must return cache_hit: False because the cache is empty.
    """
    body, _ = _post_chat(graph_id, "What is the main topic?", auth_headers)
    assert (
        body.get("cache_hit") is False
    ), f"First query must be a cache miss; got cache_hit={body.get('cache_hit')}"


# ---------------------------------------------------------------------------
# 2. Second identical query: cache_hit: True, response time <1s
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.docker
def test_second_query_cache_hit_and_fast(graph_id, auth_headers):
    """
    The second identical query must return cache_hit: True and complete in <1s
    (Redis cache is faster than live Neo4j + LLM).
    """
    query = "Who are the key entities in this graph?"

    # First call warms the cache
    _post_chat(graph_id, query, auth_headers)

    # Second call must hit the cache
    body, elapsed = _post_chat(graph_id, query, auth_headers)

    assert (
        body.get("cache_hit") is True
    ), f"Second identical query must be a cache hit; got cache_hit={body.get('cache_hit')}"
    assert elapsed < 1.0, f"Cache hit response time must be <1s; got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# 3. Ingest new document → cache invalidated → third query: cache_hit: False
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.docker
def test_cache_invalidated_after_ingest(graph_id, auth_headers):
    """
    After a new document is ingested for a graph, the cache for that graph
    must be fully invalidated.  A subsequent query must be a cache miss.
    """
    query = "Describe the knowledge in this graph."

    # Warm the cache
    _post_chat(graph_id, query, auth_headers)
    body_hit, _ = _post_chat(graph_id, query, auth_headers)
    assert (
        body_hit.get("cache_hit") is True
    ), "Pre-condition: second query must be a cache hit"

    # Ingest a new document — must trigger cache invalidation
    _post_ingest(
        graph_id,
        "New document added for testing cache invalidation.",
        auth_headers,
    )

    # Allow time for the Celery invalidation task to complete
    time.sleep(3)

    # Third query must be a cache miss (fresh result after invalidation)
    body_miss, _ = _post_chat(graph_id, query, auth_headers)
    assert (
        body_miss.get("cache_hit") is False
    ), f"After ingest, cache must be invalidated; got cache_hit={body_miss.get('cache_hit')}"


# ---------------------------------------------------------------------------
# 4. Cross-tenant isolation: graph_id A's cache not visible from graph_id B
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.docker
def test_cross_tenant_cache_isolation(auth_headers):
    """
    Cache entries for graph-A must not be visible when querying graph-B,
    even with an identical query text.  Verifies the qcache:{graph_id}: prefix
    scoping in QueryCacheService.
    """
    import httpx

    graph_id_a = _unique_graph_id()
    graph_id_b = _unique_graph_id()

    query = "What is the purpose of this graph?"

    try:
        # Warm graph-A's cache
        _post_chat(graph_id_a, query, auth_headers)
        body_a_hit, _ = _post_chat(graph_id_a, query, auth_headers)
        assert (
            body_a_hit.get("cache_hit") is True
        ), "Pre-condition: graph-A second query must be a cache hit"

        # Query graph-B with the same text — must be a cache miss (different tenant)
        body_b, _ = _post_chat(graph_id_b, query, auth_headers)
        assert (
            body_b.get("cache_hit") is False
        ), f"Graph-B must not see graph-A's cache; got cache_hit={body_b.get('cache_hit')}"

    finally:
        # Cleanup both graphs
        for gid in [graph_id_a, graph_id_b]:
            try:
                httpx.delete(f"{BASE_URL}/graphs/{gid}", timeout=10)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 5. Cache hit response includes cached answer unchanged
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.docker
def test_cache_hit_returns_same_answer(graph_id, auth_headers):
    """
    The cached response must return the same answer as the original response,
    confirming cache integrity (not just that cache_hit=True).
    """
    query = "List all entities in this graph."

    body_first, _ = _post_chat(graph_id, query, auth_headers)
    body_second, _ = _post_chat(graph_id, query, auth_headers)

    assert body_second.get("cache_hit") is True
    # The answer content must be identical
    assert body_first.get("answer") == body_second.get(
        "answer"
    ), "Cached answer must be identical to original answer"


# ---------------------------------------------------------------------------
# 6. Unique graph_id per test run (no test data collision)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unique_graph_id_format():
    """
    Verify the graph_id generator produces correctly prefixed IDs
    (no infrastructure needed — pure unit check).
    """
    gid = _unique_graph_id()
    assert gid.startswith(TEST_PREFIX)
    assert len(gid) > len(TEST_PREFIX) + 1

    # Two calls must produce different IDs
    gid2 = _unique_graph_id()
    assert gid != gid2
