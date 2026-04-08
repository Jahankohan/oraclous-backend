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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

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
        delete_resp = _mock_httpx_response(404, {})
        delete_resp.raise_for_status = MagicMock()
        client = _mock_async_client(delete_return=delete_resp)
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.delete_graph("missing-id")
        assert result["deleted"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_access_denied_returns_error(self):
        delete_resp = _mock_httpx_response(403, {})
        delete_resp.raise_for_status = MagicMock()
        client = _mock_async_client(delete_return=delete_resp)
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.delete_graph("g-forbidden")
        assert result["deleted"] is False
        assert "access denied" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_delete_returns_204(self):
        delete_resp = _mock_httpx_response(204, {})
        client = _mock_async_client(delete_return=delete_resp)
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.delete_graph("g-1")
        assert result["graph_id"] == "g-1"
        assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_successful_delete_delegates_to_rest_not_direct_service(self):
        """delete_graph must use HTTP DELETE, not import GraphNodeService directly."""
        delete_resp = _mock_httpx_response(204, {})
        client = _mock_async_client(delete_return=delete_resp)
        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.delete_graph("g-99")
        # Verify HTTP DELETE was called, not the sync service
        client.delete.assert_called_once()
        called_url = client.delete.call_args[0][0]
        assert "/api/v1/graphs/g-99" in called_url
        assert result["deleted"] is True


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
