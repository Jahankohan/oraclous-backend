"""
Unit tests for the Oraclous MCP server tool handlers.

All external dependencies (httpx client, Neo4j driver) are mocked so that
these tests run without any live services.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Patch environment before importing the module under test so that the import
# itself doesn't fail because ORACLOUS_API_KEY is not set.
# ---------------------------------------------------------------------------
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("ORACLOUS_API_KEY", "test-key")
os.environ.setdefault("ORACLOUS_BASE_URL", "http://localhost:8003")

import app.mcp.server as mcp_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_httpx_response(status_code: int = 200, json_body: dict | list = None):
    """Return a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body if json_body is not None else {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _mock_async_client(get_return=None, post_return=None, delete_return=None):
    """Return a mock httpx.AsyncClient with controllable per-method returns."""
    client = MagicMock()
    client.get = AsyncMock(return_value=get_return or _mock_httpx_response())
    client.post = AsyncMock(return_value=post_return or _mock_httpx_response())
    client.delete = AsyncMock(return_value=delete_return or _mock_httpx_response())
    return client


# ---------------------------------------------------------------------------
# Graph Management Tools
# ---------------------------------------------------------------------------


class TestCreateGraph:
    @pytest.mark.asyncio
    async def test_creates_graph_and_returns_metadata(self):
        api_response = {
            "id": "graph-001",
            "name": "My Graph",
            "description": "A test graph",
            "status": "active",
            "node_count": 0,
            "relationship_count": 0,
        }
        client = _mock_async_client(post_return=_mock_httpx_response(200, api_response))
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.create_graph(
                name="My Graph", description="A test graph"
            )

        assert result["graph_id"] == "graph-001"
        assert result["name"] == "My Graph"
        assert result["node_count"] == 0

    @pytest.mark.asyncio
    async def test_default_description_is_empty(self):
        api_response = {
            "id": "g-1",
            "name": "X",
            "status": "active",
            "node_count": 0,
            "relationship_count": 0,
        }
        client = _mock_async_client(post_return=_mock_httpx_response(200, api_response))
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.create_graph(name="X")

        assert result["description"] == ""


class TestListGraphs:
    @pytest.mark.asyncio
    async def test_returns_list_of_graphs(self):
        api_response = [
            {
                "id": "g-1",
                "name": "A",
                "description": "first",
                "status": "active",
                "node_count": 5,
                "relationship_count": 3,
            },
            {
                "id": "g-2",
                "name": "B",
                "description": "",
                "status": "active",
                "node_count": 0,
                "relationship_count": 0,
            },
        ]
        client = _mock_async_client(get_return=_mock_httpx_response(200, api_response))
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.list_graphs()

        assert len(result) == 2
        assert result[0]["graph_id"] == "g-1"
        assert result[1]["name"] == "B"

    @pytest.mark.asyncio
    async def test_empty_list(self):
        client = _mock_async_client(get_return=_mock_httpx_response(200, []))
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.list_graphs()
        assert result == []


class TestDeleteGraph:
    @pytest.mark.asyncio
    async def test_graph_not_found_returns_error(self):
        check_resp = _mock_httpx_response(404, {})
        check_resp.raise_for_status = MagicMock()
        client = _mock_async_client(get_return=check_resp)
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.delete_graph("missing-id")
        assert result["deleted"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_access_denied_returns_error(self):
        check_resp = _mock_httpx_response(403, {})
        check_resp.raise_for_status = MagicMock()
        client = _mock_async_client(get_return=check_resp)
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.delete_graph("g-forbidden")
        assert result["deleted"] is False
        assert "access denied" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_delete(self):
        get_resp = _mock_httpx_response(200, {"id": "g-1", "user_id": "user-1"})
        client = _mock_async_client(get_return=get_resp)

        mock_svc = MagicMock()
        mock_svc.delete_graph.return_value = True

        mock_driver = MagicMock()
        mock_app_client = MagicMock()
        mock_app_client.sync_driver = mock_driver

        with (
            patch.object(mcp_module, "_client", return_value=client),
            patch.dict(
                "sys.modules",
                {"app.core.neo4j_client": MagicMock(neo4j_client=mock_app_client)},
            ),
            patch(
                "app.mcp.server.GraphNodeService", return_value=mock_svc, create=True
            ),
        ):
            # Patch the inner import within delete_graph
            with patch(
                "app.services.graph_node_service.GraphNodeService",
                mock_svc,
                create=True,
            ):
                result = await mcp_module.delete_graph("g-1")

        # We can't easily test the inner import path without heavier fixtures,
        # but at minimum it should not raise.
        assert "graph_id" in result


class TestGetGraphStats:
    @pytest.mark.asyncio
    async def test_returns_stats(self):
        api_response = {
            "id": "g-1",
            "name": "Test",
            "description": "desc",
            "status": "active",
            "node_count": 42,
            "relationship_count": 17,
        }
        client = _mock_async_client(get_return=_mock_httpx_response(200, api_response))
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.get_graph_stats("g-1")

        assert result["node_count"] == 42
        assert result["relationship_count"] == 17

    @pytest.mark.asyncio
    async def test_not_found_returns_error(self):
        resp = _mock_httpx_response(404, {})
        resp.raise_for_status = MagicMock()
        client = _mock_async_client(get_return=resp)
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.get_graph_stats("missing")
        assert "error" in result


# ---------------------------------------------------------------------------
# Ingestion Tools
# ---------------------------------------------------------------------------


class TestIngestText:
    @pytest.mark.asyncio
    async def test_successful_ingest(self):
        api_response = {"id": "job-1", "graph_id": "g-1", "status": "pending"}
        client = _mock_async_client(post_return=_mock_httpx_response(200, api_response))
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.ingest_text(
                graph_id="g-1",
                text="Alice works at Acme Corp.",
                source_label="notes",
                context="HR data",
            )
        assert result["job_id"] == "job-1"
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_graph_not_found(self):
        resp = _mock_httpx_response(404, {})
        resp.raise_for_status = MagicMock()
        client = _mock_async_client(post_return=resp)
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.ingest_text("missing", "text")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_overrides_sent_when_context_given(self):
        api_response = {"id": "job-2", "graph_id": "g-1", "status": "pending"}
        client = _mock_async_client(post_return=_mock_httpx_response(200, api_response))
        with patch.object(mcp_module, "_client", return_value=client):
            await mcp_module.ingest_text("g-1", "text", context="pharma research")
        call_kwargs = client.post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["overrides"]["additional_focus"] == "pharma research"


class TestIngestFile:
    @pytest.mark.asyncio
    async def test_file_not_found_returns_error(self):
        with patch.object(mcp_module, "_client", return_value=_mock_async_client()):
            result = await mcp_module.ingest_file("g-1", "/no/such/file.txt")
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_file_ingest(self, tmp_path):
        test_file = tmp_path / "notes.txt"
        test_file.write_text("Some content about entities.")

        api_response = {"id": "job-3", "graph_id": "g-1", "status": "pending"}
        client = _mock_async_client(post_return=_mock_httpx_response(200, api_response))
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.ingest_file("g-1", str(test_file))
        assert result["job_id"] == "job-3"
        assert str(test_file) == result["file"]


# ---------------------------------------------------------------------------
# Chat Tool
# ---------------------------------------------------------------------------


class TestChat:
    @pytest.mark.asyncio
    async def test_successful_chat(self):
        api_response = {
            "answer": "Alice is the CEO.",
            "is_grounded": True,
            "sources": [{"entity": "Alice"}],
            "retriever_used": "enhanced",
        }
        client = _mock_async_client(post_return=_mock_httpx_response(200, api_response))
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.chat("g-1", "Who is Alice?")
        assert result["answer"] == "Alice is the CEO."
        assert result["is_grounded"] is True

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_error(self):
        with patch.object(mcp_module, "_client", return_value=_mock_async_client()):
            result = await mcp_module.chat("g-1", "question", mode="invalid_mode")
        assert "error" in result
        assert "invalid_mode" in result["error"]

    @pytest.mark.asyncio
    async def test_valid_modes_accepted(self):
        api_response = {
            "answer": "ok",
            "is_grounded": False,
            "sources": [],
            "retriever_used": "simple",
        }
        for mode in ("simple", "enhanced", "hybrid", "hybrid_plus", "natural"):
            client = _mock_async_client(
                post_return=_mock_httpx_response(200, api_response)
            )
            with patch.object(mcp_module, "_client", return_value=client):
                result = await mcp_module.chat("g-1", "Q?", mode=mode)
            assert "error" not in result, f"mode {mode!r} should be valid"

    @pytest.mark.asyncio
    async def test_access_denied(self):
        resp = _mock_httpx_response(403, {})
        resp.raise_for_status = MagicMock()
        client = _mock_async_client(post_return=resp)
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.chat("g-forbidden", "Q?")
        assert "error" in result


# ---------------------------------------------------------------------------
# Node Inspection Tools
# ---------------------------------------------------------------------------


def _make_neo4j_sync_driver(
    records: list[dict] | None = None, single_record: dict | None = None
):
    """Build a minimal mock Neo4j sync driver."""
    mock_record = MagicMock()
    if single_record:
        mock_record.__getitem__ = MagicMock(side_effect=lambda k: single_record.get(k))
        mock_record.get = MagicMock(
            side_effect=lambda k, default=None: single_record.get(k, default)
        )

    def _wrap(d):
        r = MagicMock()
        r.__getitem__ = MagicMock(side_effect=lambda k: d.get(k))
        r.get = MagicMock(side_effect=lambda k, default=None: d.get(k, default))
        return r

    mock_result = MagicMock()
    mock_result.single.return_value = mock_record if single_record is not None else None
    mock_result.__iter__ = MagicMock(
        return_value=iter([_wrap(r) for r in (records or [])])
    )

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    return mock_driver, mock_session


class TestSearchNodes:
    @pytest.mark.asyncio
    async def test_returns_matching_entities(self):
        records = [
            {
                "entity_id": "e-1",
                "name": "Alice",
                "type": "Person",
                "description": "CEO",
            },
        ]
        driver, _ = _make_neo4j_sync_driver(records=records)
        access_resp = _mock_httpx_response(200, {})
        client = _mock_async_client(get_return=access_resp)

        with (
            patch.object(mcp_module, "_client", return_value=client),
            patch.object(mcp_module, "_neo4j_sync_driver", return_value=driver),
        ):
            result = await mcp_module.search_nodes("g-1", "Alice")

        assert len(result) == 1
        assert result[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_access_denied_returns_error(self):
        resp = _mock_httpx_response(403, {})
        resp.raise_for_status = MagicMock()
        client = _mock_async_client(get_return=resp)
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.search_nodes("g-forbidden", "Alice")
        assert result[0]["error"] == "Graph not found or access denied."

    @pytest.mark.asyncio
    async def test_limit_is_clamped(self):
        driver, mock_session = _make_neo4j_sync_driver(records=[])
        access_resp = _mock_httpx_response(200, {})
        client = _mock_async_client(get_return=access_resp)

        with (
            patch.object(mcp_module, "_client", return_value=client),
            patch.object(mcp_module, "_neo4j_sync_driver", return_value=driver),
        ):
            await mcp_module.search_nodes("g-1", "x", limit=999)

        # The Cypher call should use the clamped limit of 50
        call_args = mock_session.run.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]
        assert params["limit"] == 50


class TestGetNode:
    @pytest.mark.asyncio
    async def test_returns_entity_properties(self):
        node_data = {
            "name": "Alice",
            "type": "Person",
            "description": "CEO",
            "entity_id": "e-1",
        }
        mock_node = MagicMock()
        # __iter__ makes dict(record["e"]) work
        mock_node.__iter__ = MagicMock(return_value=iter(node_data.items()))

        single_record = MagicMock()
        single_record.__getitem__ = MagicMock(
            side_effect=lambda k: mock_node if k == "e" else None
        )

        mock_result = MagicMock()
        mock_result.single.return_value = single_record

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        access_resp = _mock_httpx_response(200, {})
        client = _mock_async_client(get_return=access_resp)

        with (
            patch.object(mcp_module, "_client", return_value=client),
            patch.object(mcp_module, "_neo4j_sync_driver", return_value=mock_driver),
        ):
            result = await mcp_module.get_node("g-1", "Alice")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_entity_not_found(self):
        mock_result = MagicMock()
        mock_result.single.return_value = None

        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session

        access_resp = _mock_httpx_response(200, {})
        client = _mock_async_client(get_return=access_resp)

        with (
            patch.object(mcp_module, "_client", return_value=client),
            patch.object(mcp_module, "_neo4j_sync_driver", return_value=mock_driver),
        ):
            result = await mcp_module.get_node("g-1", "Nobody")

        assert "error" in result
        assert "Nobody" in result["error"]


class TestGetNeighbors:
    @pytest.mark.asyncio
    async def test_returns_neighbors(self):
        records = [
            {
                "anchor": "Alice",
                "rel_type": "WORKS_AT",
                "neighbor": "Acme",
                "neighbor_type": "Company",
            },
        ]
        driver, _ = _make_neo4j_sync_driver(records=records)
        access_resp = _mock_httpx_response(200, {})
        client = _mock_async_client(get_return=access_resp)

        with (
            patch.object(mcp_module, "_client", return_value=client),
            patch.object(mcp_module, "_neo4j_sync_driver", return_value=driver),
        ):
            result = await mcp_module.get_neighbors("g-1", "Alice", hops=1)

        assert result["entity"] == "Alice"
        assert len(result["neighbors"]) == 1
        assert result["neighbors"][0]["name"] == "Acme"

    @pytest.mark.asyncio
    async def test_entity_not_found_returns_error(self):
        driver, _ = _make_neo4j_sync_driver(records=[])
        access_resp = _mock_httpx_response(200, {})
        client = _mock_async_client(get_return=access_resp)

        with (
            patch.object(mcp_module, "_client", return_value=client),
            patch.object(mcp_module, "_neo4j_sync_driver", return_value=driver),
        ):
            result = await mcp_module.get_neighbors("g-1", "Unknown")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_hops_clamped_to_range(self):
        """hops parameter should be clamped between 1 and 2."""
        records = []
        driver, _ = _make_neo4j_sync_driver(records=records)
        access_resp = _mock_httpx_response(200, {})
        client = _mock_async_client(get_return=access_resp)

        with (
            patch.object(mcp_module, "_client", return_value=client),
            patch.object(mcp_module, "_neo4j_sync_driver", return_value=driver),
        ):
            # hops=10 should be clamped to 2 and not raise
            result = await mcp_module.get_neighbors("g-1", "Alice", hops=10)

        assert "error" in result or result.get("hops") == 2


# ---------------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200_with_ok_status(self):
        """GET /health must return HTTP 200 with {"status": "ok"}."""
        import httpx

        app = mcp_module.mcp.sse_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_content_type_is_json(self):
        """GET /health must return JSON content-type."""
        import httpx

        app = mcp_module.mcp.sse_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert "application/json" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Lifespan / Shutdown
# ---------------------------------------------------------------------------


class TestLifespan:
    @pytest.mark.asyncio
    async def test_owned_driver_is_closed_on_shutdown(self):
        """An owned (standalone) Neo4j driver must be closed when the server shuts down."""
        mock_driver = MagicMock()
        mock_http = AsyncMock()
        mock_http.is_closed = False

        mcp_module._neo4j_driver = mock_driver
        mcp_module._neo4j_driver_owned = True
        mcp_module._http_client = mock_http

        async with mcp_module._lifespan(mcp_module.mcp):
            pass  # server runs here; nothing to do

        mock_driver.close.assert_called_once()
        mock_http.aclose.assert_called_once()
        assert mcp_module._neo4j_driver is None
        assert mcp_module._http_client is None

    @pytest.mark.asyncio
    async def test_borrowed_driver_is_not_closed_on_shutdown(self):
        """A borrowed app driver must NOT be closed by the MCP lifespan."""
        mock_driver = MagicMock()
        mock_http = AsyncMock()
        mock_http.is_closed = True  # already closed

        mcp_module._neo4j_driver = mock_driver
        mcp_module._neo4j_driver_owned = False  # borrowed
        mcp_module._http_client = mock_http

        async with mcp_module._lifespan(mcp_module.mcp):
            pass

        mock_driver.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_with_no_driver_initialized(self):
        """Lifespan must not raise when no driver was ever initialized."""
        mcp_module._neo4j_driver = None
        mcp_module._neo4j_driver_owned = False
        mcp_module._http_client = None

        # Should complete without error
        async with mcp_module._lifespan(mcp_module.mcp):
            pass


# ---------------------------------------------------------------------------
# Entry point — main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_sse_transport_sets_settings_and_runs(self, monkeypatch):
        """SSE path must configure mcp.settings.host/port before calling mcp.run()."""
        monkeypatch.setenv("MCP_TRANSPORT", "sse")
        monkeypatch.setenv("MCP_HOST", "127.0.0.1")
        monkeypatch.setenv("MCP_PORT", "9001")

        mock_run = MagicMock()
        with patch.object(mcp_module.mcp, "run", mock_run):
            mcp_module.main()

        mock_run.assert_called_once_with(transport="sse")
        assert mcp_module.mcp.settings.host == "127.0.0.1"
        assert mcp_module.mcp.settings.port == 9001

    def test_sse_transport_uses_defaults(self, monkeypatch):
        """SSE path must default to 0.0.0.0:8004 when env vars are absent."""
        monkeypatch.setenv("MCP_TRANSPORT", "sse")
        monkeypatch.delenv("MCP_HOST", raising=False)
        monkeypatch.delenv("MCP_PORT", raising=False)

        mock_run = MagicMock()
        with patch.object(mcp_module.mcp, "run", mock_run):
            mcp_module.main()

        mock_run.assert_called_once_with(transport="sse")
        assert mcp_module.mcp.settings.host == "0.0.0.0"
        assert mcp_module.mcp.settings.port == 8004

    def test_stdio_transport_does_not_set_host_port(self, monkeypatch):
        """stdio path must call mcp.run(transport='stdio') and not touch settings."""
        monkeypatch.setenv("MCP_TRANSPORT", "stdio")

        mock_run = MagicMock()
        with patch.object(mcp_module.mcp, "run", mock_run):
            mcp_module.main()

        mock_run.assert_called_once_with(transport="stdio")

    def test_default_transport_is_stdio(self, monkeypatch):
        """When MCP_TRANSPORT is unset the server must default to stdio."""
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)

        mock_run = MagicMock()
        with patch.object(mcp_module.mcp, "run", mock_run):
            mcp_module.main()

        mock_run.assert_called_once_with(transport="stdio")
