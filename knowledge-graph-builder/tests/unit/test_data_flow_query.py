"""
Unit tests for the data_flow query type in code_graphs.py.

TASK-023 — QA: data_flow query dispatcher tests.
Tests the `_run_code_query` dispatcher in app/api/v1/endpoints/code_graphs.py
for the `data_flow` query type introduced by TASK-022.

Coverage:
  #1 — "data_flow" query type handled without _QueryParamError when source_symbol present
  #2 — Missing source_symbol raises _QueryParamError (400)
  #3 — Cypher references $graph_id (not hardcoded) — parameterized Cypher invariant
  #4 — data_flow forward direction returns path_nodes / path_labels / depth
  #5 — data_flow backward direction uses upstream Cypher
  #6 — data_flow "both" direction uses undirected Cypher
  #7 — Unknown query_type raises _QueryParamError
  #8 — POST /graphs/{id}/code/query with data_flow returns 200
  #9 — POST /graphs/{id}/code/query with data_flow missing source_symbol returns 400
  #10 — Multi-tenancy: $graph_id always passed in params dict (not string-interpolated)

Auth is bypassed. Neo4j is fully mocked.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

GRAPH_A = str(uuid.uuid4())
USER_A = str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — mock Neo4j records
# ─────────────────────────────────────────────────────────────────────────────


def _neo4j_record(data: dict):
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: data.get(k)
    rec.get = lambda k, default=None: data.get(k, default)
    rec.__iter__ = lambda self: iter(data)
    rec.keys = lambda: data.keys()
    rec.data = lambda: data
    return rec


def _neo4j_result(records: list[dict]):
    result = MagicMock()
    result.records = [_neo4j_record(r) for r in records]
    return result


def _run(coro):
    """Run a coroutine in the current event loop or a new one."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Test #1 — data_flow with source_symbol does not raise _QueryParamError
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_data_flow_with_source_symbol_does_not_raise():
    """
    _run_code_query with query_type='data_flow' and source_symbol present
    must not raise _QueryParamError.
    """
    from app.api.v1.endpoints.code_graphs import _QueryParamError, _run_code_query

    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([])

    with patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc:
        mnc.async_driver = mock_driver

        async def run():
            return await _run_code_query(
                graph_id=GRAPH_A,
                query_type="data_flow",
                params={"source_symbol": "mod.fn"},
                limit=50,
                include_tests=False,
            )

        result = _run(run())

    assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# Test #2 — data_flow without source_symbol raises _QueryParamError
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_data_flow_missing_source_symbol_raises_query_param_error():
    """
    _run_code_query with query_type='data_flow' and no source_symbol
    must raise _QueryParamError (translated to HTTP 400 by the endpoint).
    """
    from app.api.v1.endpoints.code_graphs import _QueryParamError, _run_code_query

    mock_driver = AsyncMock()

    with patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc:
        mnc.async_driver = mock_driver

        async def run():
            return await _run_code_query(
                graph_id=GRAPH_A,
                query_type="data_flow",
                params={},  # missing source_symbol
                limit=50,
                include_tests=False,
            )

        with pytest.raises(_QueryParamError) as exc_info:
            _run(run())

    assert "source_symbol" in str(exc_info.value).lower(), (
        f"Error message must mention 'source_symbol', got: {exc_info.value}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test #3 — Cypher references $graph_id (not hardcoded)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_data_flow_cypher_uses_graph_id_parameter():
    """
    Parameterized Cypher invariant: the Cypher executed for data_flow must
    pass graph_id as a parameter (never hardcoded string).

    Verifies that execute_query() is called with a params dict containing 'graph_id'.
    """
    from app.api.v1.endpoints.code_graphs import _run_code_query

    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([])

    with patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc:
        mnc.async_driver = mock_driver

        async def run():
            return await _run_code_query(
                graph_id=GRAPH_A,
                query_type="data_flow",
                params={"source_symbol": "mod.fn"},
                limit=50,
                include_tests=False,
            )

        _run(run())

    assert mock_driver.execute_query.called, "execute_query should have been called"

    call_args = mock_driver.execute_query.call_args
    # Second positional arg is the params dict
    if call_args.args and len(call_args.args) > 1:
        params_dict = call_args.args[1]
    elif call_args.kwargs.get("parameters"):
        params_dict = call_args.kwargs["parameters"]
    else:
        params_dict = None

    if params_dict is not None:
        assert "graph_id" in params_dict, (
            f"execute_query params must contain 'graph_id', got keys: {list(params_dict.keys())}"
        )
        assert params_dict["graph_id"] == GRAPH_A, (
            f"graph_id in params must equal the requested graph_id, "
            f"got: {params_dict['graph_id']}"
        )
    else:
        call_str = str(call_args)
        assert "graph_id" in call_str, (
            f"execute_query call must reference graph_id: {call_str}"
        )


@pytest.mark.unit
def test_data_flow_cypher_contains_graph_id_token():
    """
    The Cypher string passed to execute_query must contain '$graph_id' token —
    not a hardcoded value.
    """
    from app.api.v1.endpoints.code_graphs import _run_code_query

    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([])

    with patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc:
        mnc.async_driver = mock_driver

        async def run():
            return await _run_code_query(
                graph_id=GRAPH_A,
                query_type="data_flow",
                params={"source_symbol": "mod.fn"},
                limit=50,
                include_tests=False,
            )

        _run(run())

    call_args = mock_driver.execute_query.call_args
    cypher = call_args.args[0] if call_args.args else ""

    assert "$graph_id" in cypher, (
        f"Cypher for data_flow must use $graph_id parameter (not hardcoded). "
        f"Got Cypher: {cypher[:200]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test #4 — data_flow forward direction returns path_nodes / path_labels / depth
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_data_flow_forward_returns_path_records():
    """
    data_flow query (forward, default) returns records with:
      - path_nodes: list of qualified_names
      - path_labels: list of node labels
      - depth: path length
    """
    from app.api.v1.endpoints.code_graphs import _run_code_query

    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([
        {
            "path_nodes": ["mod.fn", "mod.fn.x", "mod.fn"],
            "path_labels": ["Function", "Variable", "Function"],
            "depth": 2,
        }
    ])

    with patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc:
        mnc.async_driver = mock_driver

        async def run():
            return await _run_code_query(
                graph_id=GRAPH_A,
                query_type="data_flow",
                params={"source_symbol": "mod.fn", "direction": "forward"},
                limit=50,
                include_tests=False,
            )

        result = _run(run())

    assert len(result) == 1
    record = result[0]
    assert "path_nodes" in record, f"Expected 'path_nodes' in result, got: {record.keys()}"
    assert "path_labels" in record, f"Expected 'path_labels' in result, got: {record.keys()}"
    assert "depth" in record, f"Expected 'depth' in result, got: {record.keys()}"
    assert record["depth"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Test #5 — data_flow backward direction uses upstream Cypher with $graph_id
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_data_flow_backward_direction_uses_graph_id():
    """
    data_flow with direction='backward' must still call execute_query
    with $graph_id in Cypher.
    """
    from app.api.v1.endpoints.code_graphs import _run_code_query

    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([])

    with patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc:
        mnc.async_driver = mock_driver

        async def run():
            return await _run_code_query(
                graph_id=GRAPH_A,
                query_type="data_flow",
                params={"source_symbol": "mod.sink", "direction": "backward"},
                limit=50,
                include_tests=False,
            )

        _run(run())

    assert mock_driver.execute_query.called
    call_args = mock_driver.execute_query.call_args
    cypher = call_args.args[0] if call_args.args else ""
    assert "$graph_id" in cypher, (
        "Backward direction Cypher must include $graph_id parameter"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test #6 — data_flow "both" direction uses undirected Cypher with $graph_id
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_data_flow_both_direction_uses_graph_id():
    """
    data_flow with direction='both' must call execute_query with $graph_id.
    """
    from app.api.v1.endpoints.code_graphs import _run_code_query

    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([])

    with patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc:
        mnc.async_driver = mock_driver

        async def run():
            return await _run_code_query(
                graph_id=GRAPH_A,
                query_type="data_flow",
                params={"source_symbol": "mod.node", "direction": "both"},
                limit=50,
                include_tests=False,
            )

        _run(run())

    assert mock_driver.execute_query.called
    call_args = mock_driver.execute_query.call_args
    cypher = call_args.args[0] if call_args.args else ""
    assert "$graph_id" in cypher, (
        "Both-direction Cypher must include $graph_id parameter"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test #7 — Unknown query_type raises _QueryParamError
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_unknown_query_type_raises_error():
    """Unknown query_type (not in the registered types) raises _QueryParamError."""
    from app.api.v1.endpoints.code_graphs import _QueryParamError, _run_code_query

    mock_driver = AsyncMock()

    with patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc:
        mnc.async_driver = mock_driver

        async def run():
            return await _run_code_query(
                graph_id=GRAPH_A,
                query_type="nonexistent_type",
                params={},
                limit=50,
                include_tests=False,
            )

        with pytest.raises(_QueryParamError):
            _run(run())


# ─────────────────────────────────────────────────────────────────────────────
# Test #8 — POST /graphs/{id}/code/query with data_flow returns 200
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_code_query_data_flow_endpoint_returns_200():
    """
    POST /graphs/{graph_id}/code/query with query_type=data_flow and
    source_symbol present must return HTTP 200 with correct shape.

    Uses a minimal FastAPI app that mounts only the code_graphs router.
    Patches auth_service.verify_token to bypass the real auth stack.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.v1.endpoints.code_graphs import router as code_graphs_router

    minimal_app = FastAPI()
    minimal_app.include_router(code_graphs_router, prefix="/api/v1")

    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([
        {
            "path_nodes": ["views.handle_request", "views.handle_request.req"],
            "path_labels": ["Function", "Variable"],
            "depth": 1,
        }
    ])

    fake_user = {"id": USER_A, "email": "test@test.com", "principal_type": "user"}

    with (
        # Patch at the endpoint module's namespace (not the source module)
        # because code_graphs.py imports neo4j_client with `from ... import`
        patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc,
        patch(
            "app.services.auth_service.auth_service.verify_token",
            new=AsyncMock(return_value=fake_user),
        ),
        patch(
            "app.api.v1.endpoints.code_graphs.verify_graph_access",
            new=AsyncMock(),
        ),
    ):
        mnc.async_driver = mock_driver

        with TestClient(minimal_app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/graphs/{GRAPH_A}/code/query",
                headers={"Authorization": "Bearer test"},
                json={
                    "query_type": "data_flow",
                    "params": {"source_symbol": "views.handle_request"},
                },
            )

    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body["query_type"] == "data_flow"
    assert body["total"] == 1
    assert len(body["results"]) == 1
    assert "path_nodes" in body["results"][0]


# ─────────────────────────────────────────────────────────────────────────────
# Test #9 — POST /graphs/{id}/code/query with data_flow missing source_symbol → 400
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_code_query_data_flow_endpoint_missing_source_symbol_returns_400():
    """
    POST /graphs/{graph_id}/code/query with query_type=data_flow but no
    source_symbol must return HTTP 400.

    Uses minimal FastAPI app + auth_service.verify_token mock.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.v1.endpoints.code_graphs import router as code_graphs_router

    minimal_app = FastAPI()
    minimal_app.include_router(code_graphs_router, prefix="/api/v1")

    mock_driver = AsyncMock()
    fake_user = {"id": USER_A, "email": "test@test.com", "principal_type": "user"}

    with (
        patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc,
        patch(
            "app.services.auth_service.auth_service.verify_token",
            new=AsyncMock(return_value=fake_user),
        ),
        patch(
            "app.api.v1.endpoints.code_graphs.verify_graph_access",
            new=AsyncMock(),
        ),
    ):
        mnc.async_driver = mock_driver

        with TestClient(minimal_app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/graphs/{GRAPH_A}/code/query",
                headers={"Authorization": "Bearer test"},
                json={
                    "query_type": "data_flow",
                    "params": {},  # missing source_symbol
                },
            )

    assert resp.status_code == 400, (
        f"Expected 400 for missing source_symbol, got {resp.status_code}: {resp.text}"
    )
    detail = resp.json().get("detail", "")
    assert "source_symbol" in detail.lower(), (
        f"Error detail must mention 'source_symbol', got: {detail}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test #10 — Multi-tenancy: graph_id never hardcoded in Cypher string
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_data_flow_graph_id_never_hardcoded_in_cypher_string():
    """
    The Cypher string passed to execute_query for data_flow must NEVER contain
    the literal graph_id UUID — it must always be passed as a parameter.
    """
    from app.api.v1.endpoints.code_graphs import _run_code_query

    test_graph_id = "fixed-tenant-uuid-9999"

    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([])

    with patch("app.api.v1.endpoints.code_graphs.neo4j_client") as mnc:
        mnc.async_driver = mock_driver

        async def run():
            return await _run_code_query(
                graph_id=test_graph_id,
                query_type="data_flow",
                params={"source_symbol": "mod.fn"},
                limit=50,
                include_tests=False,
            )

        _run(run())

    call_args = mock_driver.execute_query.call_args
    cypher = call_args.args[0] if call_args.args else ""

    # The literal UUID must NOT appear inside the Cypher string
    assert test_graph_id not in cypher, (
        f"Graph ID '{test_graph_id}' must not be hardcoded in Cypher: {cypher[:300]}"
    )

    # But it must appear in the parameters dict
    if len(call_args.args) > 1:
        params_dict = call_args.args[1]
        assert params_dict.get("graph_id") == test_graph_id, (
            f"graph_id must be passed as parameter, got: {params_dict}"
        )
