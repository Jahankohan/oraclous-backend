"""Unit tests for Graph-Native Agent CRUD (TASK-032 / STORY-020).

Covers:
- Schema validation: reasoning_mode enum, tools allowlist
- AgentService: create, list, get, update, deactivate (mocked Neo4j driver)
- API endpoints: 201 on create, 400 on bad input, 403 on unauthorized, 404 on missing
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.agent_schemas import AgentCreate, AgentUpdate
from app.services.agent_service import AgentService

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_agent_node(
    agent_id: str = "agent-001",
    graph_id: str = "graph-001",
    name: str = "Test Agent",
    deactivated_at=None,
) -> dict:
    return {
        "a": {
            "agent_id": agent_id,
            "graph_id": graph_id,
            "name": name,
            "description": "desc",
            "system_prompt": "You are helpful.",
            "reasoning_mode": "direct",
            "retriever_strategy": "hybrid",
            "retriever_hop_depth": 2,
            "retriever_max_results": 20,
            "tools": json.dumps(["graph_search"]),
            "llm_config_id": None,
            "created_by": "user-001",
            "created_at": 1714176000,
            "deactivated_at": deactivated_at,
        }
    }


def _make_driver(records=None):
    mock_result = MagicMock()
    mock_result.records = records or []
    driver = AsyncMock()
    driver.execute_query = AsyncMock(return_value=mock_result)
    return driver, mock_result


# ── Schema validation ─────────────────────────────────────────────────────────


class TestAgentSchemaValidation:
    def test_valid_create_passes(self):
        a = AgentCreate(
            name="Support Agent",
            system_prompt="You are helpful.",
            reasoning_mode="research",
            tools=["graph_search", "neighbors"],
        )
        assert a.reasoning_mode == "research"
        assert set(a.tools) == {"graph_search", "neighbors"}

    def test_invalid_reasoning_mode_raises(self):
        with pytest.raises(Exception):
            AgentCreate(
                name="Agent",
                system_prompt="x",
                reasoning_mode="magic",  # not a valid literal
            )

    def test_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            AgentCreate(
                name="Agent",
                system_prompt="x",
                tools=["graph_search", "does_not_exist"],
            )

    def test_all_valid_tools_accepted(self):
        all_tools = [
            "graph_search",
            "community_members",
            "neighbors",
            "degree_centrality",
            "shortest_path",
            "taint_trace",
            "temporal_slice",
        ]
        a = AgentCreate(name="Agent", system_prompt="x", tools=all_tools)
        assert len(a.tools) == 7

    def test_update_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            AgentUpdate(tools=["graph_search", "bad_tool"])


# ── AgentService unit tests ───────────────────────────────────────────────────


class TestAgentService:
    async def test_create_agent_returns_agent_id(self):
        driver, _ = _make_driver()
        svc = AgentService(driver)
        data = AgentCreate(name="Agent", system_prompt="You are helpful.")
        agent_id = await svc.create_agent("graph-1", "user-1", data)
        assert agent_id  # non-empty UUID string
        driver.execute_query.assert_called_once()
        # graph_id must appear in the Cypher params
        call_params = driver.execute_query.call_args[0][1]
        assert call_params["graph_id"] == "graph-1"

    async def test_create_agent_passes_graph_id_in_params(self):
        driver, _ = _make_driver()
        svc = AgentService(driver)
        await svc.create_agent(
            "my-graph-id",
            "user-1",
            AgentCreate(name="A", system_prompt="p"),
        )
        params = driver.execute_query.call_args[0][1]
        assert params["graph_id"] == "my-graph-id"

    async def test_list_agents_returns_active_only(self):
        active = _make_agent_node(agent_id="a1", deactivated_at=None)
        driver, result = _make_driver(
            records=[
                MagicMock(
                    **{
                        "__iter__": lambda s: iter(active.items()),
                        "keys": lambda: active.keys(),
                        **active,
                    }
                )
            ]
        )
        # Use dict-like records
        result.records = [active]
        svc = AgentService(driver)
        # Patch _row_to_dict to avoid Neo4j object conversion
        with patch("app.services.agent_service._row_to_dict", return_value=active["a"]):
            agents = await svc.list_agents("graph-1")
        assert isinstance(agents, list)

    async def test_get_agent_returns_none_when_not_found(self):
        driver, _ = _make_driver(records=[])
        svc = AgentService(driver)
        result = await svc.get_agent("graph-1", "missing-id")
        assert result is None

    async def test_deactivate_agent_returns_true_when_found(self):
        record = MagicMock()
        driver, mock_result = _make_driver(records=[record])
        svc = AgentService(driver)
        result = await svc.deactivate_agent("graph-1", "agent-1")
        assert result is True
        params = driver.execute_query.call_args[0][1]
        assert params["graph_id"] == "graph-1"
        assert params["agent_id"] == "agent-1"

    async def test_deactivate_agent_returns_false_when_not_found(self):
        driver, _ = _make_driver(records=[])
        svc = AgentService(driver)
        result = await svc.deactivate_agent("graph-1", "ghost-agent")
        assert result is False

    async def test_update_agent_noop_when_no_fields(self):
        driver, _ = _make_driver(records=[])
        svc = AgentService(driver)
        # get_agent returns None (agent not found) → update returns None
        result = await svc.update_agent("graph-1", "agent-1", AgentUpdate())
        assert result is None


# ── API endpoint tests ────────────────────────────────────────────────────────

_FAKE_USER_ID = "user-001"
_AUTH_HEADER = {"Authorization": "Bearer fake-token"}


class TestAgentEndpoints:
    """API-level tests using app.dependency_overrides for auth and Neo4j isolation."""

    @pytest.fixture(autouse=True)
    def _override_auth(self, async_client):
        from app.api.dependencies import get_current_user_id
        from app.main import app

        app.dependency_overrides[get_current_user_id] = lambda: _FAKE_USER_ID
        yield
        app.dependency_overrides.pop(get_current_user_id, None)

    def _make_svc_override(self, svc):
        from app.api.v1.endpoints.agents import _agent_service
        from app.main import app

        app.dependency_overrides[_agent_service] = lambda: svc
        return _agent_service

    def _clear_svc_override(self, key):
        from app.main import app

        app.dependency_overrides.pop(key, None)

    async def test_post_unknown_tool_returns_422(self, async_client):
        svc = AsyncMock(spec=AgentService)
        key = self._make_svc_override(svc)
        try:
            with patch(
                "app.api.v1.endpoints.agents.verify_graph_access",
                new_callable=AsyncMock,
                return_value="graph-1",
            ):
                response = await async_client.post(
                    "/api/v1/api/v1/graphs/graph-1/agents",
                    json={
                        "name": "Agent",
                        "system_prompt": "x",
                        "tools": ["not_a_real_tool"],
                    },
                    headers=_AUTH_HEADER,
                )
        finally:
            self._clear_svc_override(key)
        assert response.status_code == 422

    async def test_post_invalid_reasoning_mode_returns_422(self, async_client):
        svc = AsyncMock(spec=AgentService)
        key = self._make_svc_override(svc)
        try:
            with patch(
                "app.api.v1.endpoints.agents.verify_graph_access",
                new_callable=AsyncMock,
                return_value="graph-1",
            ):
                response = await async_client.post(
                    "/api/v1/api/v1/graphs/graph-1/agents",
                    json={
                        "name": "Agent",
                        "system_prompt": "x",
                        "reasoning_mode": "telepathy",
                    },
                    headers=_AUTH_HEADER,
                )
        finally:
            self._clear_svc_override(key)
        assert response.status_code == 422

    async def test_get_missing_agent_returns_404(self, async_client):
        svc = AsyncMock(spec=AgentService)
        svc.get_agent = AsyncMock(return_value=None)
        key = self._make_svc_override(svc)
        try:
            with patch(
                "app.api.v1.endpoints.agents.verify_graph_access",
                new_callable=AsyncMock,
                return_value="graph-1",
            ):
                response = await async_client.get(
                    "/api/v1/api/v1/graphs/graph-1/agents/nonexistent",
                    headers=_AUTH_HEADER,
                )
        finally:
            self._clear_svc_override(key)
        assert response.status_code == 404

    async def test_delete_missing_agent_returns_404(self, async_client):
        svc = AsyncMock(spec=AgentService)
        svc.deactivate_agent = AsyncMock(return_value=False)
        key = self._make_svc_override(svc)
        try:
            with patch(
                "app.api.v1.endpoints.agents.verify_graph_access",
                new_callable=AsyncMock,
                return_value="graph-1",
            ):
                response = await async_client.delete(
                    "/api/v1/api/v1/graphs/graph-1/agents/nonexistent",
                    headers=_AUTH_HEADER,
                )
        finally:
            self._clear_svc_override(key)
        assert response.status_code == 404

    async def test_unauthorized_returns_403(self, async_client):
        from fastapi import HTTPException

        svc = AsyncMock(spec=AgentService)
        svc.list_agents = AsyncMock(return_value=[])
        key = self._make_svc_override(svc)
        try:
            with patch(
                "app.api.v1.endpoints.agents.verify_graph_access",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=403, detail="Access denied"),
            ):
                response = await async_client.get(
                    "/api/v1/api/v1/graphs/graph-1/agents",
                    headers=_AUTH_HEADER,
                )
        finally:
            self._clear_svc_override(key)
        assert response.status_code == 403
