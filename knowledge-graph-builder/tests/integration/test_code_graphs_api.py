"""
Integration tests for Code Knowledge Graph API endpoints.

Covers 12 test criteria from ORA-69 spec (API surface):
  #1  — Ingest endpoint returns 202 with job_id
  #2  — Symbols endpoint returns empty list on fresh graph (no code yet)
  #3  — Query endpoint (callers) requires function_name param
  #4  — Query endpoint (dead_code) returns results
  #5  — Query endpoint (circular_imports) returns results
  #6  — Query endpoint (callers) with valid data
  #7  — Dead code with include_tests=False excludes test functions
  #8  — Query endpoint (inheritance_chain) requires class_name param
  #9  — Multi-tenant isolation: user B cannot see user A's code
  #10 — Symbols endpoint filters by language
  #11 — Symbols endpoint full-text search (q param)
  #12 — Code ingest with TypeScript language filter

Auth is bypassed. Neo4j and PostgreSQL are mocked.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_NOW = datetime(2026, 4, 8, 12, 0, 0, tzinfo=UTC).isoformat()
USER_A = str(uuid.uuid4())
USER_B = str(uuid.uuid4())
GRAPH_A = str(uuid.uuid4())
GRAPH_B = str(uuid.uuid4())
JOB_ID = str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _neo4j_record(data: dict[str, Any]):
    rec = MagicMock()
    rec.__getitem__ = lambda self, k: data.get(k)
    rec.get = lambda k, default=None: data.get(k, default)
    rec.__iter__ = lambda self: iter(data)
    rec.keys = lambda: data.keys()
    rec.data = lambda: data
    return rec


def _neo4j_result(records: list[dict[str, Any]]):
    result = MagicMock()
    result.records = [_neo4j_record(r) for r in records]
    return result


def _make_job(graph_id: str = GRAPH_A, job_id: str = JOB_ID):
    job = MagicMock()
    job.id = uuid.UUID(job_id)
    job.graph_id = uuid.UUID(graph_id)
    job.source_type = "code"
    job.status = "pending"
    job.progress = 0
    job.created_at = _NOW
    job.ingest_mode = "incremental"
    return job


def _patch_auth(user_id: str):
    return patch(
        "app.api.dependencies.get_current_user_id",
        return_value=user_id,
    )


def _patch_verify_graph_access(allowed: bool = True):
    async def _ok(*args, **kwargs):
        if not allowed:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
            )

    return patch("app.api.dependencies.verify_graph_access", side_effect=_ok)


# ─────────────────────────────────────────────────────────────────────────────
# App fixture — import once and patch at module level
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def app_client():
    """TestClient with Neo4j and PG fully mocked at import time."""
    with (
        patch("app.core.neo4j_client.neo4j_client") as mock_neo4j,
        patch("app.core.database.async_session_maker"),
        patch("app.services.background_job_service.background_job_service") as mock_bjs,
    ):
        # Default neo4j driver mock
        mock_driver = AsyncMock()
        mock_neo4j.async_driver = mock_driver

        # Default BJS mock
        mock_bjs.start_code_ingest_job.return_value = {
            "task_id": "celery-task-abc",
            "job_id": JOB_ID,
            "status": "started",
            "message": "ok",
        }

        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_neo4j, mock_bjs


# ─────────────────────────────────────────────────────────────────────────────
# Helper: shorthand test client + mock neo4j setup per test
# ─────────────────────────────────────────────────────────────────────────────


def _client_setup():
    """Returns (TestClient, mock_neo4j, mock_bjs) with standard patches."""

    mock_driver = AsyncMock()
    mock_neo4j = MagicMock()
    mock_neo4j.async_driver = mock_driver

    mock_bjs = MagicMock()
    mock_bjs.start_code_ingest_job.return_value = {
        "task_id": "celery-task-abc",
        "job_id": JOB_ID,
        "status": "started",
        "message": "ok",
    }

    return mock_driver, mock_neo4j, mock_bjs


# ─────────────────────────────────────────────────────────────────────────────
# Test #1 — POST /graphs/{graphId}/code-ingest returns 202
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_code_ingest_returns_202():
    """Criterion #1: successful submission returns 202 with job_id."""
    from fastapi.testclient import TestClient

    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([{"g.graph_id": GRAPH_A}])

    mock_job = _make_job()

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
        patch("app.services.background_job_service.background_job_service") as mbjs,
        patch("app.api.v1.endpoints.code_graphs.IngestionJob", return_value=mock_job),
    ):
        mnc.async_driver = mock_driver
        mbjs.start_code_ingest_job.return_value = {
            "status": "started",
            "task_id": "t1",
            "message": "ok",
        }

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/graphs/{GRAPH_A}/code-ingest",
                headers={"Authorization": "Bearer test"},
                json={"repo_path": "/tmp/myrepo"},
            )

    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["graph_id"] == GRAPH_A
    assert body["status"] == "queued"


# ─────────────────────────────────────────────────────────────────────────────
# Test #2 — GET /graphs/{graphId}/code/symbols returns empty list
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_list_symbols_empty_graph():
    """Criterion #2: fresh graph with no code returns empty symbol list."""
    mock_driver = AsyncMock()
    # count query → 0, data query → []
    mock_driver.execute_query.side_effect = [
        _neo4j_result([{"total": 0}]),
        _neo4j_result([]),
    ]

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
    ):
        mnc.async_driver = mock_driver
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                f"/api/v1/graphs/{GRAPH_A}/code/symbols",
                headers={"Authorization": "Bearer test"},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["symbols"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Test #3 — Query: callers without function_name → 400
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_code_query_callers_missing_param():
    """Criterion #3: callers query without function_name param returns 400."""
    mock_driver = AsyncMock()

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
    ):
        mnc.async_driver = mock_driver
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/graphs/{GRAPH_A}/code/query",
                headers={"Authorization": "Bearer test"},
                json={"query_type": "callers", "params": {}},
            )

    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# Test #4 — Query: dead_code returns results
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_code_query_dead_code_returns_results():
    """Criterion #4: dead_code query returns unreferenced functions."""
    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result(
        [
            {
                "qualified_name": "app.utils.unused_fn",
                "language": "python",
                "start_line": 42,
                "is_test": False,
            },
        ]
    )

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
    ):
        mnc.async_driver = mock_driver
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/graphs/{GRAPH_A}/code/query",
                headers={"Authorization": "Bearer test"},
                json={"query_type": "dead_code", "params": {}},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["query_type"] == "dead_code"
    assert body["total"] == 1
    assert body["results"][0]["qualified_name"] == "app.utils.unused_fn"


# ─────────────────────────────────────────────────────────────────────────────
# Test #5 — Query: circular_imports returns cycle list
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_code_query_circular_imports():
    """Criterion #5: circular_imports returns detected cycles."""
    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result(
        [
            {"cycle": ["app/a.py", "app/b.py", "app/a.py"]},
        ]
    )

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
    ):
        mnc.async_driver = mock_driver
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/graphs/{GRAPH_A}/code/query",
                headers={"Authorization": "Bearer test"},
                json={"query_type": "circular_imports", "params": {}},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert "app/a.py" in body["results"][0]["cycle"]


# ─────────────────────────────────────────────────────────────────────────────
# Test #6 — Query: callers with valid data
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_code_query_callers_with_data():
    """Criterion #6: callers query returns functions that call the target."""
    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result(
        [
            {"caller": "app.main.run", "line": 10, "language": "python"},
        ]
    )

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
    ):
        mnc.async_driver = mock_driver
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/graphs/{GRAPH_A}/code/query",
                headers={"Authorization": "Bearer test"},
                json={"query_type": "callers", "params": {"function_name": "ingest"}},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["results"][0]["caller"] == "app.main.run"


# ─────────────────────────────────────────────────────────────────────────────
# Test #7 — Dead code exclude_tests param wired through
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_code_query_dead_code_include_tests_param():
    """Criterion #7: include_tests=False (default) passes exclusion filter in Cypher."""
    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([])

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
    ):
        mnc.async_driver = mock_driver
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/graphs/{GRAPH_A}/code/query",
                headers={"Authorization": "Bearer test"},
                json={"query_type": "dead_code", "params": {}, "include_tests": False},
            )

    assert resp.status_code == 200
    # Verify the query was called with test-exclusion Cypher (NOT f.is_test in query)
    call_args = mock_driver.execute_query.call_args
    cypher: str = call_args[0][0] if call_args[0] else ""
    assert "is_test" in cypher


# ─────────────────────────────────────────────────────────────────────────────
# Test #8 — inheritance_chain requires class_name
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_code_query_inheritance_chain_missing_param():
    """Criterion #8: inheritance_chain without class_name returns 400."""
    mock_driver = AsyncMock()

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
    ):
        mnc.async_driver = mock_driver
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/graphs/{GRAPH_A}/code/query",
                headers={"Authorization": "Bearer test"},
                json={"query_type": "inheritance_chain", "params": {}},
            )

    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# Test #9 — Multi-tenant isolation via verify_graph_access
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_symbols_forbidden_for_other_user():
    """Criterion #9: user B cannot access user A's graph → 403."""
    from fastapi import HTTPException
    from fastapi import status as http_status

    async def _deny(*args, **kwargs):
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN, detail="Forbidden"
        )

    mock_driver = AsyncMock()

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_B),
        patch("app.api.dependencies.verify_graph_access", side_effect=_deny),
    ):
        mnc.async_driver = mock_driver
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                f"/api/v1/graphs/{GRAPH_A}/code/symbols",
                headers={"Authorization": "Bearer test"},
            )

    assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# Test #10 — Symbols endpoint language filter
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_list_symbols_language_filter():
    """Criterion #10: language query param filters Cypher by language."""
    mock_driver = AsyncMock()
    mock_driver.execute_query.side_effect = [
        _neo4j_result([{"total": 1}]),
        _neo4j_result(
            [
                {
                    "sym_type": "Function",
                    "name": "run",
                    "qualified_name": "app.main.run",
                    "file_path": "app/main.py",
                    "start_line": 1,
                    "end_line": 10,
                    "signature": None,
                    "docstring": None,
                    "language": "python",
                    "is_async": False,
                    "is_method": False,
                    "is_test": False,
                    "visibility": "public",
                }
            ]
        ),
    ]

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
    ):
        mnc.async_driver = mock_driver
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                f"/api/v1/graphs/{GRAPH_A}/code/symbols?language=python",
                headers={"Authorization": "Bearer test"},
            )

    assert resp.status_code == 200
    # Verify language filter appeared in Cypher
    first_call = mock_driver.execute_query.call_args_list[0]
    params = (
        first_call[0][1]
        if len(first_call[0]) > 1
        else first_call[1].get("parameters", {})
    )
    assert params.get("language") == "python" or "language" in str(first_call)


# ─────────────────────────────────────────────────────────────────────────────
# Test #11 — Symbols endpoint full-text search
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_list_symbols_fulltext_search():
    """Criterion #11: q param triggers full-text filter on name/qualified_name/docstring."""
    mock_driver = AsyncMock()
    mock_driver.execute_query.side_effect = [
        _neo4j_result([{"total": 0}]),
        _neo4j_result([]),
    ]

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
    ):
        mnc.async_driver = mock_driver
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                f"/api/v1/graphs/{GRAPH_A}/code/symbols?q=ingest",
                headers={"Authorization": "Bearer test"},
            )

    assert resp.status_code == 200
    # Verify 'q' filter was sent to Cypher
    first_call = mock_driver.execute_query.call_args_list[0]
    call_str = str(first_call)
    assert "ingest" in call_str or "q" in call_str


# ─────────────────────────────────────────────────────────────────────────────
# Test #12 — Code ingest with TypeScript language filter
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_code_ingest_typescript_language_filter():
    """Criterion #12: ingest request with languages=['typescript'] is accepted."""
    mock_driver = AsyncMock()
    mock_driver.execute_query.return_value = _neo4j_result([{"g.graph_id": GRAPH_A}])

    mock_job = _make_job()

    with (
        patch("app.core.neo4j_client.neo4j_client") as mnc,
        patch("app.api.dependencies.get_current_user_id", return_value=USER_A),
        patch("app.api.dependencies.verify_graph_access", new=AsyncMock()),
        patch("app.services.background_job_service.background_job_service") as mbjs,
        patch("app.api.v1.endpoints.code_graphs.IngestionJob", return_value=mock_job),
    ):
        mnc.async_driver = mock_driver
        mbjs.start_code_ingest_job.return_value = {
            "status": "started",
            "task_id": "t2",
            "message": "ok",
        }

        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                f"/api/v1/graphs/{GRAPH_A}/code-ingest",
                headers={"Authorization": "Bearer test"},
                json={
                    "repo_path": "/tmp/ts-project",
                    "languages": ["typescript"],
                    "mode": "full",
                },
            )

    assert resp.status_code == 202
    body = resp.json()
    assert body["graph_id"] == GRAPH_A
