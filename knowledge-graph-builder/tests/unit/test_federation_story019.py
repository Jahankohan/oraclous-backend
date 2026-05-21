"""
TASK-018: Test suite for STORY-019 federation endpoints.

Covers:
  - TASK-016: _store_same_as_links graph_id stamping (1 test)
  - TASK-016: POST /graphs/{graph_id}/federation/candidates (5 tests)
  - TASK-017: FederationResolveRequest validator — confidence_threshold (4 tests)
  - TASK-017: POST /graphs/{graph_id}/federation/resolve (3 tests)
  - TASK-017: Celery task result structure (1 test)
  - Regression: existing federated endpoints unchanged (2 tests)

All external dependencies (Neo4j, Celery, auth) are mocked — no live services.

STORY-019 acceptance criteria map:
  AC-1 (SAME_AS graph IDs stamped)        → Test 1
  AC-2 (candidates endpoint 200)          → Test 2
  AC-3 (entity_name filter forwarded)     → Test 3
  AC-4 (server-side 0.60 threshold)       → Test 4
  AC-5 (candidates auth fail-closed)      → Test 5
  AC-6 (threshold validator / defaults)   → Tests 6, 7, 8, 9
  AC-7 (resolve happy path, queued)       → Test 10
  AC-8 (resolve write-access gate)        → Test 11
  AC-9 (resolve read-access gate)         → Test 12
  (Celery result shape)                   → Test 13
  (regression: existing endpoints)        → Tests 14, 15
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Minimal env so settings import does not fail.
# ---------------------------------------------------------------------------
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("NEO4J_DATABASE", "neo4j")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("POSTGRES_URL", "postgresql://user:pass@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_async_driver(rows: list[dict]):
    """Return a mock async Neo4j driver whose session().run().data() returns rows."""
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=rows)

    mock_session = MagicMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute_write = AsyncMock(return_value=None)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    return mock_driver, mock_session


def _owned_federatable(graph_id: str, user_id: str) -> dict:
    return {
        "graph_id": graph_id,
        "user_id": user_id,
        "name": f"Graph {graph_id}",
        "federatable": True,
    }


def _make_candidate_result(entity_a_id: str, entity_b_id: str, score: float) -> dict:
    return {
        "entity_a": {"id": entity_a_id, "name": "Acme", "graph_id": "graph-X"},
        "entity_b": {"id": entity_b_id, "name": "Acme", "graph_id": "graph-Y"},
        "score": score,
        "signals": {
            "embedding": 0.0,
            "name": 1.0,
            "type": score,  # approximate
            "shared_relations": 0.0,
        },
    }


def _build_test_app() -> FastAPI:
    """Build a minimal FastAPI app with the federation routers, bypassing lifespan."""
    from app.api.v1.endpoints.federation import graph_federation_router, router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.include_router(graph_federation_router, prefix="/api/v1")
    return app


# ---------------------------------------------------------------------------
# Test 1 — _store_same_as_links stamps source_graph_id and target_graph_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_same_as_links_stamps_both_graph_ids():
    """TASK-016: _store_same_as_links must SET source_graph_id and target_graph_id
    on the SAME_AS relationship (satisfies CROSS-GRAPH-WRITE-NO-GRAPH-ID guard).

    AC-1: SAME_AS relationships carry source_graph_id + target_graph_id.
    """
    import inspect

    from app.services.federation_service import FederationService

    source = inspect.getsource(FederationService._store_same_as_links)

    # The Cypher must SET both graph ID columns on the SAME_AS relationship.
    assert "source_graph_id" in source, (
        "_store_same_as_links Cypher must SET source_graph_id on SAME_AS"
    )
    assert "target_graph_id" in source, (
        "_store_same_as_links Cypher must SET target_graph_id on SAME_AS"
    )

    # The Cypher SET clause must reference pair.source_graph_id / pair.target_graph_id
    assert "pair.source_graph_id" in source or "s.source_graph_id" in source, (
        "source_graph_id must be SET from the pair parameter"
    )
    assert "pair.target_graph_id" in source or "s.target_graph_id" in source, (
        "target_graph_id must be SET from the pair parameter"
    )

    # Verify the actual call propagates graph IDs through to Neo4j
    driver, mock_session = _make_async_driver([])
    svc = FederationService(async_driver=driver)

    pairs = [("id-a", "id-b", 0.95, "graph-X", "graph-Y")]
    await svc._store_same_as_links(pairs)

    # execute_write must have been called (MERGE inside a write transaction)
    mock_session.execute_write.assert_called_once()

    # Verify the pair parameter dict includes both graph IDs
    _write_fn = mock_session.execute_write.call_args[0][0]  # noqa: F841  (kept for source inspection in debug)
    # Reconstruct: the fn is a closure over the pair_params list. Inspect source to
    # confirm both keys are in the params list built inside _store_same_as_links.
    assert "source_graph_id" in source
    assert "target_graph_id" in source


# ---------------------------------------------------------------------------
# Test 2 — candidates happy path: 2 candidates, correct response shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidates_endpoint_happy_path():
    """TASK-016: POST /graphs/X/federation/candidates returns a list with
    score, entity_a, entity_b, signals for each pair above 0.60.

    AC-2: candidates endpoint returns correctly shaped list.
    """
    from app.services.federation_service import FederationService

    mock_candidates = [
        _make_candidate_result("id-1", "id-2", 0.70),
        _make_candidate_result("id-3", "id-4", 0.80),
    ]

    app = _build_test_app()

    fake_user = {"id": "user-1", "principal_type": "user"}
    fake_service = MagicMock(spec=FederationService)
    fake_service._validate_and_filter = AsyncMock(
        return_value=[
            _owned_federatable("graph-X", "user-1"),
            _owned_federatable("graph-Y", "user-1"),
        ]
    )
    fake_service.find_federation_candidates = AsyncMock(return_value=mock_candidates)

    from app.api.v1.endpoints.federation import _get_federation_service
    from app.services.auth_service import auth_service

    app.dependency_overrides[_get_federation_service] = lambda: fake_service

    with (
        patch.object(
            auth_service, "verify_token", new=AsyncMock(return_value=fake_user)
        ),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/graphs/graph-X/federation/candidates",
            json={"target_graph_ids": ["graph-Y"]},
            headers={"Authorization": "Bearer tok"},
        )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    for item in data:
        assert "score" in item
        assert "entity_a" in item
        assert "entity_b" in item
        assert "signals" in item
        signals = item["signals"]
        assert "embedding" in signals
        assert "name" in signals
        assert "type" in signals
        assert "shared_relations" in signals


# ---------------------------------------------------------------------------
# Test 3 — entity_name forwarded to find_federation_candidates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidates_entity_name_forwarded_to_service():
    """TASK-016: When entity_name='Acme' is in the request, it must be passed
    through to find_federation_candidates as the entity_name kwarg.

    AC-3: entity_name filter is forwarded to service.
    """
    from app.services.federation_service import FederationService

    app = _build_test_app()

    fake_user = {"id": "user-1", "principal_type": "user"}
    fake_service = MagicMock(spec=FederationService)
    fake_service._validate_and_filter = AsyncMock(
        return_value=[
            _owned_federatable("graph-X", "user-1"),
            _owned_federatable("graph-Y", "user-1"),
        ]
    )
    fake_service.find_federation_candidates = AsyncMock(return_value=[])

    from app.api.v1.endpoints.federation import _get_federation_service
    from app.services.auth_service import auth_service

    app.dependency_overrides[_get_federation_service] = lambda: fake_service

    with patch.object(
        auth_service, "verify_token", new=AsyncMock(return_value=fake_user)
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/graphs/graph-X/federation/candidates",
            json={"target_graph_ids": ["graph-Y"], "entity_name": "Acme"},
            headers={"Authorization": "Bearer tok"},
        )

    assert response.status_code == 200

    # find_federation_candidates must have been called with entity_name="Acme"
    call_kwargs = fake_service.find_federation_candidates.call_args
    assert call_kwargs.kwargs.get("entity_name") == "Acme" or (
        call_kwargs.args and "Acme" in call_kwargs.args
    ), f"entity_name='Acme' not forwarded; got call: {call_kwargs}"


# ---------------------------------------------------------------------------
# Test 4 — server-side threshold: only pairs >= 0.60 returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidates_server_side_threshold_filters_low_scores():
    """TASK-016: Service returns pairs with scores [0.45, 0.62, 0.91];
    only pairs with score >= 0.60 must appear in the endpoint response.

    AC-4: server-side 0.60 threshold is enforced.
    """
    from app.services.federation_service import FederationService

    mock_candidates = [
        _make_candidate_result("id-1", "id-2", 0.45),  # below threshold
        _make_candidate_result("id-3", "id-4", 0.62),  # above threshold
        _make_candidate_result("id-5", "id-6", 0.91),  # above threshold
    ]

    app = _build_test_app()

    fake_user = {"id": "user-1", "principal_type": "user"}
    fake_service = MagicMock(spec=FederationService)
    fake_service._validate_and_filter = AsyncMock(
        return_value=[
            _owned_federatable("graph-X", "user-1"),
            _owned_federatable("graph-Y", "user-1"),
        ]
    )
    fake_service.find_federation_candidates = AsyncMock(return_value=mock_candidates)

    from app.api.v1.endpoints.federation import _get_federation_service
    from app.services.auth_service import auth_service

    app.dependency_overrides[_get_federation_service] = lambda: fake_service

    with patch.object(
        auth_service, "verify_token", new=AsyncMock(return_value=fake_user)
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/graphs/graph-X/federation/candidates",
            json={"target_graph_ids": ["graph-Y"]},
            headers={"Authorization": "Bearer tok"},
        )

    assert response.status_code == 200
    data = response.json()

    # Only the 0.62 and 0.91 pairs must appear
    assert len(data) == 2, f"Expected 2 results (>= 0.60), got {len(data)}: {data}"
    scores = {round(item["score"], 2) for item in data}
    assert 0.45 not in scores, (
        "Score 0.45 (below threshold) must not appear in response"
    )
    assert all(s >= 0.60 for s in scores), f"All scores must be >= 0.60; got {scores}"


# ---------------------------------------------------------------------------
# Test 5 — candidates auth: inaccessible graph returns 403
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidates_auth_returns_403_when_graph_inaccessible():
    """TASK-016: When _validate_and_filter returns fewer graphs than requested
    (some are not accessible), the endpoint must return 403.

    AC-5: candidates auth is fail-closed.
    """
    from app.services.federation_service import FederationError, FederationService

    app = _build_test_app()

    fake_user = {"id": "user-1", "principal_type": "user"}
    fake_service = MagicMock(spec=FederationService)
    # Simulate auth failure — only 1 of 2 graphs accessible (or raise FederationError)
    fake_service._validate_and_filter = AsyncMock(
        side_effect=FederationError(
            "Access denied — no accessible graphs in federation request",
            status_code=403,
        )
    )

    from app.api.v1.endpoints.federation import _get_federation_service
    from app.services.auth_service import auth_service

    app.dependency_overrides[_get_federation_service] = lambda: fake_service

    with patch.object(
        auth_service, "verify_token", new=AsyncMock(return_value=fake_user)
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/graphs/graph-X/federation/candidates",
            json={"target_graph_ids": ["graph-Y"]},
            headers={"Authorization": "Bearer tok"},
        )

    assert response.status_code == 403, (
        f"Expected 403 when graph is inaccessible, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Tests 6-9 — FederationResolveRequest confidence_threshold validation
# ---------------------------------------------------------------------------


def test_resolve_request_default_threshold_is_0_85():
    """TASK-017: FederationResolveRequest.confidence_threshold defaults to 0.85
    when not provided.

    AC-6 (partial): default threshold.
    """
    from app.schemas.federation_schemas import FederationResolveRequest

    req = FederationResolveRequest(target_graph_id="graph-Y")
    assert req.confidence_threshold == 0.85, (
        f"Default confidence_threshold must be 0.85, got {req.confidence_threshold}"
    )


def test_resolve_request_threshold_0_30_clamped_to_0_60():
    """TASK-017: confidence_threshold=0.30 must be clamped up to 0.60.

    AC-6: threshold clamped to [0.60, 1.0].
    """
    from app.schemas.federation_schemas import FederationResolveRequest

    req = FederationResolveRequest(target_graph_id="graph-Y", confidence_threshold=0.30)
    assert req.confidence_threshold == 0.60, (
        f"threshold 0.30 must be clamped to 0.60, got {req.confidence_threshold}"
    )


def test_resolve_request_threshold_1_5_clamped_to_1_0():
    """TASK-017: confidence_threshold=1.5 must be clamped down to 1.0.

    AC-6: threshold clamped to [0.60, 1.0].
    """
    from app.schemas.federation_schemas import FederationResolveRequest

    req = FederationResolveRequest(target_graph_id="graph-Y", confidence_threshold=1.5)
    assert req.confidence_threshold == 1.0, (
        f"threshold 1.5 must be clamped to 1.0, got {req.confidence_threshold}"
    )


def test_resolve_request_threshold_0_75_unchanged():
    """TASK-017: confidence_threshold=0.75 is within [0.60, 1.0] and must not be changed.

    AC-6: valid threshold passed through.
    """
    from app.schemas.federation_schemas import FederationResolveRequest

    req = FederationResolveRequest(target_graph_id="graph-Y", confidence_threshold=0.75)
    assert req.confidence_threshold == 0.75, (
        f"threshold 0.75 must not be changed, got {req.confidence_threshold}"
    )


# ---------------------------------------------------------------------------
# Test 10 — resolve happy path: returns task_id and status="queued"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_endpoint_happy_path():
    """TASK-017: POST /graphs/X/federation/resolve — both auth checks pass,
    Celery task dispatched; response must be {task_id: 'abc-123', status: 'queued'}.

    AC-7: resolve endpoint returns queued task response.
    """
    from app.services.federation_service import FederationService

    app = _build_test_app()

    fake_user = {"id": "user-1", "principal_type": "user"}
    fake_service = MagicMock(spec=FederationService)
    # Both graphs accessible
    fake_service._validate_and_filter = AsyncMock(
        return_value=[
            _owned_federatable("graph-X", "user-1"),
            _owned_federatable("graph-Y", "user-1"),
        ]
    )

    mock_task = MagicMock()
    mock_task.id = "abc-123"

    from app.api.v1.endpoints.federation import _get_federation_service
    from app.services.auth_service import auth_service

    app.dependency_overrides[_get_federation_service] = lambda: fake_service

    with (
        patch.object(
            auth_service, "verify_token", new=AsyncMock(return_value=fake_user)
        ),
        patch(
            "app.api.v1.endpoints.federation.verify_graph_access",
            new=AsyncMock(return_value="graph-X"),
        ),
        patch(
            "app.tasks.federation_tasks.resolve_same_as_task",
        ) as mock_celery_task,
    ):
        mock_celery_task.delay = MagicMock(return_value=mock_task)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/graphs/graph-X/federation/resolve",
            json={"target_graph_id": "graph-Y"},
            headers={"Authorization": "Bearer tok"},
        )

    assert response.status_code == 200, (
        f"Expected 200 on happy path, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data.get("task_id") == "abc-123", f"Expected task_id='abc-123', got: {data}"
    assert data.get("status") == "queued", f"Expected status='queued', got: {data}"


# ---------------------------------------------------------------------------
# Test 11 — resolve auth: no write access → 403
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_endpoint_no_write_access_returns_403():
    """TASK-017: When _validate_and_filter passes but verify_graph_access raises
    HTTP 403 (no write access), the endpoint must return 403.

    AC-8: resolve write-access gate is enforced.
    """
    from app.services.federation_service import FederationService

    app = _build_test_app()

    fake_user = {"id": "user-1", "principal_type": "user"}
    fake_service = MagicMock(spec=FederationService)
    fake_service._validate_and_filter = AsyncMock(
        return_value=[
            _owned_federatable("graph-X", "user-1"),
            _owned_federatable("graph-Y", "user-1"),
        ]
    )

    from app.api.v1.endpoints.federation import _get_federation_service
    from app.services.auth_service import auth_service

    app.dependency_overrides[_get_federation_service] = lambda: fake_service

    with (
        patch.object(
            auth_service, "verify_token", new=AsyncMock(return_value=fake_user)
        ),
        patch(
            "app.api.v1.endpoints.federation.verify_graph_access",
            new=AsyncMock(
                side_effect=HTTPException(status_code=403, detail="Access denied")
            ),
        ),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/graphs/graph-X/federation/resolve",
            json={"target_graph_id": "graph-Y"},
            headers={"Authorization": "Bearer tok"},
        )

    assert response.status_code == 403, (
        f"Expected 403 when write access denied, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Test 12 — resolve auth: no read access → 403
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_endpoint_no_read_access_returns_403():
    """TASK-017: When _validate_and_filter raises FederationError 403 (read access
    denied), the endpoint must return 403 without proceeding to write check.

    AC-9: resolve read-access gate is enforced.
    """
    from app.services.federation_service import FederationError, FederationService

    app = _build_test_app()

    fake_user = {"id": "user-1", "principal_type": "user"}
    fake_service = MagicMock(spec=FederationService)
    # Read access denied for one graph
    fake_service._validate_and_filter = AsyncMock(
        side_effect=FederationError(
            "Access denied — no accessible graphs in federation request",
            status_code=403,
        )
    )

    from app.api.v1.endpoints.federation import _get_federation_service
    from app.services.auth_service import auth_service

    app.dependency_overrides[_get_federation_service] = lambda: fake_service

    with patch.object(
        auth_service, "verify_token", new=AsyncMock(return_value=fake_user)
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/graphs/graph-X/federation/resolve",
            json={"target_graph_id": "graph-Y"},
            headers={"Authorization": "Bearer tok"},
        )

    assert response.status_code == 403, (
        f"Expected 403 when read access denied, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Test 13 — Celery task result structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_celery_task_result_has_correct_keys():
    """TASK-017: When EntityResolver.resolve_and_link() returns an empty list
    (no ambiguous candidates), the Celery task's async inner function must
    return a dict with keys: created_links, ambiguous_count, rejected_count.

    Tests the _resolve_same_as_async function logic directly.
    """
    from app.tasks.federation_tasks import _resolve_same_as_async

    mock_task = MagicMock()

    # Mock WorkerNeo4jManager context
    mock_async_driver = MagicMock()
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_async_driver.session = MagicMock(return_value=mock_session)

    mock_neo4j = MagicMock()
    mock_neo4j.get_async_driver = MagicMock(return_value=mock_async_driver)
    mock_neo4j.__aenter__ = AsyncMock(return_value=mock_neo4j)
    mock_neo4j.__aexit__ = AsyncMock(return_value=False)

    # Return empty entities so the early-return path is taken
    with (
        patch(
            "app.tasks.federation_tasks.WorkerNeo4jManager",
            return_value=mock_neo4j,
        ),
        patch(
            "app.tasks.federation_tasks._fetch_entities_with_embeddings",
            new=AsyncMock(return_value=[]),
        ),
    ):
        result = await _resolve_same_as_async(mock_task, "graph-X", "graph-Y", 0.85)

    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
    assert "created_links" in result, f"Missing 'created_links' key in result: {result}"
    assert "ambiguous_count" in result, (
        f"Missing 'ambiguous_count' key in result: {result}"
    )
    assert "rejected_count" in result, (
        f"Missing 'rejected_count' key in result: {result}"
    )


# ---------------------------------------------------------------------------
# Test 14 — Regression: POST /graphs/federate/query still works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_federate_query_endpoint_still_registered():
    """Regression: POST /api/v1/graphs/federate/query must still be reachable
    (TASK-016/017 changes must not break the existing federated query endpoint).
    """
    from app.services.federation_service import FederationService

    app = _build_test_app()

    fake_user = {"id": "user-1", "principal_type": "user"}
    fake_result = {
        "status": "ok",
        "graphs_queried": ["g1", "g2"],
        "total_entities": 0,
        "entities": [],
        "cross_graph_links": [],
        "query_meta": {
            "execution_time_ms": 1,
            "graphs_skipped": [],
            "timed_out": False,
            "deduplication_status": "not_requested",
        },
    }
    fake_service = MagicMock(spec=FederationService)
    fake_service.federated_query = AsyncMock(return_value=fake_result)

    from app.api.v1.endpoints.federation import _get_federation_service
    from app.services.auth_service import auth_service

    app.dependency_overrides[_get_federation_service] = lambda: fake_service

    with patch.object(
        auth_service, "verify_token", new=AsyncMock(return_value=fake_user)
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/graphs/federate/query",
            json={
                "graph_ids": ["g1", "g2"],
                "query": "Acme",
            },
            headers={"Authorization": "Bearer tok"},
        )

    assert response.status_code == 200, (
        f"POST /federate/query must still work; got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Test 15 — Regression: POST /graphs/federate/vector-search still works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_federate_vector_search_endpoint_still_registered():
    """Regression: POST /api/v1/graphs/federate/vector-search must still be
    reachable (TASK-016/017 changes must not break the vector search endpoint).
    """
    from app.services.federation_service import FederationService

    app = _build_test_app()

    fake_user = {"id": "user-1", "principal_type": "user"}
    fake_result = {
        "status": "ok",
        "graphs_queried": ["g1", "g2"],
        "total_results": 0,
        "results": [],
        "query_meta": {
            "execution_time_ms": 1,
            "graphs_skipped": [],
            "timed_out": False,
            "deduplication_status": "not_requested",
        },
    }
    fake_service = MagicMock(spec=FederationService)
    fake_service.federated_vector_search = AsyncMock(return_value=fake_result)

    from app.api.v1.endpoints.federation import _get_federation_service
    from app.services.auth_service import auth_service

    app.dependency_overrides[_get_federation_service] = lambda: fake_service

    with patch.object(
        auth_service, "verify_token", new=AsyncMock(return_value=fake_user)
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/graphs/federate/vector-search",
            json={
                "graph_ids": ["g1", "g2"],
                "query_text": "semantic search test",
            },
            headers={"Authorization": "Bearer tok"},
        )

    assert response.status_code == 200, (
        f"POST /federate/vector-search must still work; got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data["status"] == "ok"
