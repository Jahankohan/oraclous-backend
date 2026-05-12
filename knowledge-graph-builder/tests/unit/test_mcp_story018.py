"""
TASK-015: Test suite for STORY-018 MCP tools.

Covers:
  - TASK-013: detect_communities, list_communities, get_community (8 scenarios)
  - TASK-014: chat tool temporal + retriever_type extensions (5 scenarios)
  - Regression: total tool count assertion (1 scenario)

All external dependencies (httpx client) are mocked — no live services required.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Set env before importing the module so the import does not fail on missing key.
# ---------------------------------------------------------------------------
os.environ.setdefault("ORACLOUS_API_KEY", "test-key")
os.environ.setdefault("ORACLOUS_BASE_URL", "http://localhost:8003")

import app.mcp.server as mcp_module  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — mirrors the pattern used in test_mcp_server.py
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8003"


def _mock_response(
    status_code: int = 200, json_body: dict | list | None = None, *, text: str = ""
):
    """Return a mock httpx.Response for the community tools (uses is_success)."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    _body = json_body if json_body is not None else {}
    resp.json.return_value = _body
    resp.text = text or str(_body)
    # raise_for_status is used by chat / create_graph / etc. (not community tools)
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _mock_client(get_return=None, post_return=None):
    """Return a mock httpx.AsyncClient."""
    client = MagicMock()
    client.get = AsyncMock(return_value=get_return or _mock_response())
    client.post = AsyncMock(return_value=post_return or _mock_response())
    return client


# ===========================================================================
# TASK-013 — Community Detection Tools
# ===========================================================================


class TestDetectCommunities:
    """Tests for the detect_communities MCP tool (TASK-013)."""

    @pytest.mark.asyncio
    async def test_happy_path_correct_url_and_auth(self):
        """detect_communities: POST to /api/v1/graphs/{graph_id}/communities/detect
        with auth header forwarded; response includes job_id."""
        job_response = {"job_id": "job-abc", "status": "pending", "graph_id": "g-1"}
        client = _mock_client(post_return=_mock_response(200, job_response))

        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.detect_communities(graph_id="g-1")

        # URL must include the expected path
        call_kwargs = client.post.call_args
        url = (
            call_kwargs.args[0]
            if call_kwargs.args
            else call_kwargs.kwargs.get("url", "")
        )
        assert "/api/v1/graphs/g-1/communities/detect" in url

        # Auth header must be forwarded
        headers = call_kwargs.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer test-key"

        # Response must contain job_id
        assert "job_id" in result
        assert result["job_id"] == "job-abc"

    @pytest.mark.asyncio
    async def test_with_resolutions_body_contains_resolutions(self):
        """detect_communities: when resolutions=[0.5, 1.0], the POST body
        must include {"resolutions": [0.5, 1.0]}."""
        job_response = {"job_id": "job-xyz", "status": "pending", "graph_id": "g-1"}
        client = _mock_client(post_return=_mock_response(200, job_response))

        with patch.object(mcp_module, "_client", return_value=client):
            await mcp_module.detect_communities(graph_id="g-1", resolutions=[0.5, 1.0])

        call_kwargs = client.post.call_args
        body = call_kwargs.kwargs.get("json", {})
        assert "resolutions" in body
        assert body["resolutions"] == [0.5, 1.0]

    @pytest.mark.asyncio
    async def test_without_resolutions_body_is_empty(self):
        """detect_communities: when no resolutions are passed, POST body must be empty {}."""
        job_response = {"job_id": "job-1", "status": "pending", "graph_id": "g-1"}
        client = _mock_client(post_return=_mock_response(200, job_response))

        with patch.object(mcp_module, "_client", return_value=client):
            await mcp_module.detect_communities(graph_id="g-1")

        call_kwargs = client.post.call_args
        body = call_kwargs.kwargs.get("json", {})
        assert "resolutions" not in body

    @pytest.mark.asyncio
    async def test_rest_error_503_returns_structured_error(self):
        """detect_communities: on HTTP 503 the tool must return
        {"error": ..., "status_code": 503} — no unhandled exception raised."""
        error_body = {"detail": "Service temporarily unavailable"}
        error_resp = _mock_response(503, error_body)
        client = _mock_client(post_return=error_resp)

        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.detect_communities(graph_id="g-1")

        assert "error" in result
        assert result["status_code"] == 503
        # Must NOT raise — result is a dict, not an exception


class TestListCommunities:
    """Tests for the list_communities MCP tool (TASK-013)."""

    @pytest.mark.asyncio
    async def test_happy_path_with_level_query_params(self):
        """list_communities: GET /api/v1/graphs/{graph_id}/communities with
        query params level=2&include_summary=true."""
        communities_response = {"items": [], "total": 0}
        client = _mock_client(get_return=_mock_response(200, communities_response))

        with patch.object(mcp_module, "_client", return_value=client):
            await mcp_module.list_communities(graph_id="X", level=2)

        call_kwargs = client.get.call_args
        url = (
            call_kwargs.args[0]
            if call_kwargs.args
            else call_kwargs.kwargs.get("url", "")
        )
        params = call_kwargs.kwargs.get("params", {})

        assert "/api/v1/graphs/X/communities" in url
        assert str(params.get("level")) == "2"
        assert str(params.get("include_summary")) == "true"

    @pytest.mark.asyncio
    async def test_without_level_omits_level_param(self):
        """list_communities: when level is not supplied, include_summary=true
        must still be present but 'level' must NOT appear in query params."""
        communities_response = {"items": [], "total": 0}
        client = _mock_client(get_return=_mock_response(200, communities_response))

        with patch.object(mcp_module, "_client", return_value=client):
            await mcp_module.list_communities(graph_id="X")

        call_kwargs = client.get.call_args
        params = call_kwargs.kwargs.get("params", {})

        assert "include_summary" in params
        assert str(params["include_summary"]) == "true"
        assert "level" not in params

    @pytest.mark.asyncio
    async def test_rest_error_404_returns_structured_error(self):
        """list_communities: on HTTP 404 the tool must return a structured
        error dict containing 'error' and 'status_code': 404."""
        error_body = {"detail": "Graph not found"}
        error_resp = _mock_response(404, error_body)
        client = _mock_client(get_return=error_resp)

        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.list_communities(graph_id="missing")

        assert "error" in result
        assert result["status_code"] == 404


class TestGetCommunity:
    """Tests for the get_community MCP tool (TASK-013)."""

    @pytest.mark.asyncio
    async def test_happy_path_url_contains_community_id(self):
        """get_community: GET URL must contain /communities/C1."""
        community_response = {
            "id": "C1",
            "level": 1,
            "summary": "Tech companies",
            "members": [],
        }
        client = _mock_client(get_return=_mock_response(200, community_response))

        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.get_community(graph_id="X", community_id="C1")

        call_kwargs = client.get.call_args
        url = (
            call_kwargs.args[0]
            if call_kwargs.args
            else call_kwargs.kwargs.get("url", "")
        )
        assert "/communities/C1" in url

        # Result is the raw JSON from the API
        assert result["id"] == "C1"

    @pytest.mark.asyncio
    async def test_rest_error_500_returns_structured_error(self):
        """get_community: on HTTP 500 the tool must return a structured error
        dict containing 'error' and 'status_code': 500."""
        error_body = {"detail": "Internal server error"}
        error_resp = _mock_response(500, error_body)
        client = _mock_client(get_return=error_resp)

        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.get_community(graph_id="X", community_id="C1")

        assert "error" in result
        assert result["status_code"] == 500


# ===========================================================================
# TASK-014 — Chat Tool Temporal + Retriever Extensions
# ===========================================================================


class TestChatExtensions:
    """Tests for the extended chat MCP tool (TASK-014)."""

    # ------------------------------------------------------------------
    # Helper: build a standard successful chat API response
    # ------------------------------------------------------------------
    @staticmethod
    def _chat_api_response():
        return {
            "answer": "The answer.",
            "is_grounded": True,
            "sources": [],
            "retriever_used": "enhanced",
        }

    def _successful_client(self):
        return _mock_client(post_return=_mock_response(200, self._chat_api_response()))

    # ------------------------------------------------------------------
    # Scenario 9: retriever_type included in body
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_with_retriever_type_body_contains_retriever_type_and_mode(self):
        """chat: when retriever_type='community_summary' is supplied, the POST body
        must contain retriever_type AND the default mode='enhanced'."""
        client = self._successful_client()

        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.chat(
                graph_id="X",
                question="q",
                retriever_type="community_summary",
            )

        assert "error" not in result
        call_kwargs = client.post.call_args
        body = call_kwargs.kwargs.get("json", {})
        assert body.get("retriever_type") == "community_summary"
        assert body.get("mode") == "enhanced"

    # ------------------------------------------------------------------
    # Scenario 10: temporal_mode=point_in_time
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_temporal_mode_point_in_time_includes_temporal_at(self):
        """chat: temporal_mode='point_in_time' + temporal_at must both appear in body."""
        client = self._successful_client()

        with patch.object(mcp_module, "_client", return_value=client):
            await mcp_module.chat(
                graph_id="X",
                question="q",
                temporal_mode="point_in_time",
                temporal_at="2023-06-01T00:00:00Z",
            )

        call_kwargs = client.post.call_args
        body = call_kwargs.kwargs.get("json", {})
        assert body.get("temporal_mode") == "point_in_time"
        assert body.get("temporal_at") == "2023-06-01T00:00:00Z"

    # ------------------------------------------------------------------
    # Scenario 11: temporal_mode=changes_since
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_temporal_mode_changes_since_includes_temporal_since_not_temporal_at(
        self,
    ):
        """chat: temporal_mode='changes_since' + temporal_since in body;
        temporal_at must NOT be present."""
        client = self._successful_client()

        with patch.object(mcp_module, "_client", return_value=client):
            await mcp_module.chat(
                graph_id="X",
                question="q",
                temporal_mode="changes_since",
                temporal_since="2026-01-01T00:00:00Z",
            )

        call_kwargs = client.post.call_args
        body = call_kwargs.kwargs.get("json", {})
        assert body.get("temporal_mode") == "changes_since"
        assert body.get("temporal_since") == "2026-01-01T00:00:00Z"
        assert "temporal_at" not in body

    # ------------------------------------------------------------------
    # Scenario 12: None params omitted from body
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_none_params_omitted_from_body(self):
        """chat: None-valued params (retriever_type, temporal_mode, temporal_at,
        temporal_since) must NOT appear as keys in the POST body."""
        client = self._successful_client()

        with patch.object(mcp_module, "_client", return_value=client):
            await mcp_module.chat(
                graph_id="X",
                question="q",
                retriever_type=None,
                temporal_mode=None,
                temporal_at=None,
                temporal_since=None,
            )

        call_kwargs = client.post.call_args
        body = call_kwargs.kwargs.get("json", {})
        assert "retriever_type" not in body
        assert "temporal_mode" not in body
        assert "temporal_at" not in body
        assert "temporal_since" not in body

    # ------------------------------------------------------------------
    # Scenario 13: existing default behavior unchanged
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_default_call_produces_pre_task014_body(self):
        """chat: default call (no new params) must produce the exact same body
        as before TASK-014: {query, graph_id, mode, include_sources} only."""
        client = self._successful_client()

        with patch.object(mcp_module, "_client", return_value=client):
            result = await mcp_module.chat(graph_id="X", question="q")

        assert "error" not in result
        call_kwargs = client.post.call_args
        body = call_kwargs.kwargs.get("json", {})

        # Must match the pre-TASK-014 body exactly
        expected_body = {
            "query": "q",
            "graph_id": "X",
            "mode": "enhanced",
            "include_sources": True,
        }
        assert body == expected_body


# ===========================================================================
# Regression — Tool Count
# ===========================================================================


class TestToolCount:
    """Regression: total registered MCP tool count never regresses below the STORY-018 floor of 13.
    SPRINT-002 added 25 more (21 assessment + 4 registry); future stories add more still."""

    @pytest.mark.asyncio
    async def test_total_tool_count_at_least_13(self):
        """The MCP server must expose at least the 13 STORY-018-era tools (10 original + 3 community
        from TASK-013). SPRINT-002 brought the running total to 38 and future sprints will add more,
        so this assertion is a floor, not an equality."""
        tools = await mcp_module.mcp.list_tools()
        tool_names = [t.name for t in tools]

        assert len(tools) >= 13, (
            f"Expected at least 13 tools (STORY-018 floor), got {len(tools)}. "
            f"Tools: {tool_names}"
        )

    @pytest.mark.asyncio
    async def test_community_tools_are_registered(self):
        """The three community tools from TASK-013 must be present in the tool list."""
        tools = await mcp_module.mcp.list_tools()
        tool_names = {t.name for t in tools}

        assert "detect_communities" in tool_names
        assert "list_communities" in tool_names
        assert "get_community" in tool_names

    @pytest.mark.asyncio
    async def test_chat_tool_is_registered(self):
        """The chat tool must still be registered (TASK-014 does not replace it)."""
        tools = await mcp_module.mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert "chat" in tool_names

    @pytest.mark.asyncio
    async def test_original_tools_still_present(self):
        """All 10 original tools must still be registered after STORY-018 changes."""
        tools = await mcp_module.mcp.list_tools()
        tool_names = {t.name for t in tools}

        original_tools = {
            "create_graph",
            "list_graphs",
            "delete_graph",
            "get_graph_stats",
            "ingest_text",
            "ingest_file",
            "chat",
            "search_nodes",
            "get_node",
            "get_neighbors",
        }
        missing = original_tools - tool_names
        assert not missing, f"Original tools missing from registry: {missing}"
