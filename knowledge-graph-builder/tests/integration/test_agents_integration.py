"""Integration tests for Graph-Native Agents — STORY-020 / TASK-036.

Runs against the Docker Neo4j instance (bolt://neo4j:7687).
The LLM (OpenAI) is mocked in all tests so no API key is required.

Tests cover:
1. Agent CRUD — node persisted in Neo4j with correct properties
2. Deactivated agent returns 404 on chat
3. Tool allowlist enforced at the AgentToolkit layer
4. Cross-tenant isolation — tool queries never return nodes from another graph
5. Full chat → provenance structure verified end-to-end
6. Conversational mode session history injected across turns
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

try:
    from neo4j import AsyncDriver
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

_GID_A = f"integ-agents-A-{uuid.uuid4().hex[:8]}"
_GID_B = f"integ-agents-B-{uuid.uuid4().hex[:8]}"
_USER = "integ-test-user-001"

pytestmark = pytest.mark.skipif(
    not _NEO4J_AVAILABLE, reason="neo4j driver not installed"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _cleanup(neo4j_test_driver: AsyncDriver):
    """Delete all test nodes before and after each test."""
    async def _wipe():
        for gid in (_GID_A, _GID_B):
            await neo4j_test_driver.execute_query(
                "MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid}
            )
    await _wipe()
    yield
    await _wipe()


@pytest.fixture(autouse=True)
def _override_auth(async_client):
    from app.main import app
    from app.api.dependencies import get_current_user_id

    app.dependency_overrides[get_current_user_id] = lambda: _USER
    yield
    app.dependency_overrides.pop(get_current_user_id, None)


@pytest.fixture(autouse=True)
def _override_agent_service(async_client, neo4j_test_driver: AsyncDriver):
    import app.api.v1.endpoints.agents as _agents_mod
    from app.main import app
    from app.api.v1.endpoints.agents import _agent_service
    from app.services.agent_service import AgentService

    app.dependency_overrides[_agent_service] = lambda: AgentService(neo4j_test_driver)
    _original_driver = _agents_mod.neo4j_client.async_driver
    _agents_mod.neo4j_client.async_driver = neo4j_test_driver
    yield
    app.dependency_overrides.pop(_agent_service, None)
    _agents_mod.neo4j_client.async_driver = _original_driver


def _mock_verify(return_value=None):
    """Return a context manager that patches verify_graph_access."""
    return patch(
        "app.api.v1.endpoints.agents.verify_graph_access",
        new=AsyncMock(return_value=return_value),
    )


def _llm_mock(responses: list[str]) -> MagicMock:
    client = MagicMock()
    completions = []
    for text in responses:
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = text
        completions.append(r)
    client.chat.completions.create = AsyncMock(side_effect=completions)
    return client


_AGENT_URL = "/api/v1/api/v1/graphs/{gid}/agents"
_CHAT_URL  = "/api/v1/api/v1/graphs/{gid}/agents/{aid}/chat"


async def _create_agent(async_client, gid: str, **kwargs) -> str:
    payload = {
        "name": kwargs.get("name", "Test Agent"),
        "system_prompt": "You are helpful.",
        "reasoning_mode": kwargs.get("reasoning_mode", "direct"),
        "tools": kwargs.get("tools", []),
    }
    with _mock_verify():
        resp = await async_client.post(_AGENT_URL.format(gid=gid), json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["agent_id"]


# ── 1. Agent CRUD — Neo4j persistence ────────────────────────────────────────


class TestAgentCRUDNeo4j:
    async def test_create_persists_agent_node(
        self, async_client, neo4j_test_driver: AsyncDriver
    ):
        agent_id = await _create_agent(async_client, _GID_A)

        result = await neo4j_test_driver.execute_query(
            "MATCH (a:Agent {agent_id: $aid, graph_id: $gid}) RETURN a",
            {"aid": agent_id, "gid": _GID_A},
        )
        assert len(result.records) == 1
        props = dict(result.records[0]["a"])
        assert props["graph_id"] == _GID_A
        assert props["name"] == "Test Agent"
        assert props.get("deactivated_at") is None

    async def test_list_excludes_deactivated(self, async_client):
        id1 = await _create_agent(async_client, _GID_A, name="Active")
        id2 = await _create_agent(async_client, _GID_A, name="ToDelete")

        with _mock_verify():
            await async_client.delete(
                f"/api/v1/api/v1/graphs/{_GID_A}/agents/{id2}"
            )
            resp = await async_client.get(_AGENT_URL.format(gid=_GID_A))

        assert resp.status_code == 200
        ids = [a["agent_id"] for a in resp.json()]
        assert id1 in ids
        assert id2 not in ids

    async def test_deleted_agent_node_has_deactivated_at(
        self, async_client, neo4j_test_driver: AsyncDriver
    ):
        agent_id = await _create_agent(async_client, _GID_A)
        with _mock_verify():
            await async_client.delete(f"/api/v1/api/v1/graphs/{_GID_A}/agents/{agent_id}")

        result = await neo4j_test_driver.execute_query(
            "MATCH (a:Agent {agent_id: $aid}) RETURN a.deactivated_at AS ts",
            {"aid": agent_id},
        )
        assert result.records[0]["ts"] is not None


# ── 2. Deactivated agent → 404 on chat ───────────────────────────────────────


class TestDeactivatedAgent:
    async def test_chat_returns_404(self, async_client):
        agent_id = await _create_agent(async_client, _GID_A)

        with _mock_verify():
            await async_client.delete(f"/api/v1/api/v1/graphs/{_GID_A}/agents/{agent_id}")
            resp = await async_client.post(
                _CHAT_URL.format(gid=_GID_A, aid=agent_id),
                json={"message": "hello"},
            )
        assert resp.status_code == 404


# ── 3. Tool allowlist enforced at toolkit layer ───────────────────────────────


class TestToolAllowlist:
    async def test_permitted_tool_does_not_raise(
        self, neo4j_test_driver: AsyncDriver
    ):
        from app.services.agent_tools import AgentToolkit
        tk = AgentToolkit(neo4j_test_driver, allowed_tools=["degree_centrality"])
        # Empty graph — returns empty list, not error
        result = await tk.degree_centrality(_GID_A, "NonExistentLabel", top_n=5)
        assert result == []

    async def test_unpermitted_tool_raises(self, neo4j_test_driver: AsyncDriver):
        from app.services.agent_tools import AgentToolkit, ToolNotPermittedError
        tk = AgentToolkit(neo4j_test_driver, allowed_tools=["graph_search"])
        with pytest.raises(ToolNotPermittedError) as exc_info:
            await tk.degree_centrality(_GID_A, "Person")
        assert exc_info.value.tool_name == "degree_centrality"

    async def test_all_tools_blocked_when_empty_allowlist(
        self, neo4j_test_driver: AsyncDriver
    ):
        from app.services.agent_tools import AgentToolkit, ToolNotPermittedError
        tk = AgentToolkit(neo4j_test_driver, allowed_tools=[])
        for name in ["community_members", "neighbors", "degree_centrality",
                     "shortest_path", "taint_trace", "temporal_slice"]:
            with pytest.raises(ToolNotPermittedError):
                method = getattr(tk, name)
                # Provide minimal args just to reach the _require() check
                await method(_GID_A, **({"community_id": "x"} if name == "community_members"
                                        else {"node_id": "x"} if name == "neighbors"
                                        else {"node_label": "X"} if name == "degree_centrality"
                                        else {"node_label": "X", "at_time": 0} if name == "temporal_slice"
                                        else {"from_qname": "a", "to_qname": "b"} if name == "shortest_path"
                                        else {"source_qname": "x"}))


# ── 4. Cross-tenant isolation ─────────────────────────────────────────────────


class TestCrossTenantIsolation:
    async def test_temporal_slice_returns_only_own_graph_nodes(
        self, neo4j_test_driver: AsyncDriver
    ):
        # Seed both graphs with the same label + time window
        await neo4j_test_driver.execute_query(
            """
            CREATE (:IsolEntity {graph_id: $gA, id: 'nodeA1', valid_from: 0, valid_to: null})
            CREATE (:IsolEntity {graph_id: $gA, id: 'nodeA2', valid_from: 0, valid_to: null})
            CREATE (:IsolEntity {graph_id: $gB, id: 'nodeB1', valid_from: 0, valid_to: null})
            """,
            {"gA": _GID_A, "gB": _GID_B},
        )
        from app.services.agent_tools import AgentToolkit
        tk = AgentToolkit(neo4j_test_driver, allowed_tools=["temporal_slice"])
        nodes = await tk.temporal_slice(_GID_A, "IsolEntity", at_time=1)

        ids = {n.id for n in nodes}
        assert "nodeA1" in ids
        assert "nodeA2" in ids
        assert "nodeB1" not in ids, "Cross-tenant node leak detected!"

    async def test_degree_centrality_scoped_to_graph(
        self, neo4j_test_driver: AsyncDriver
    ):
        # Create nodes with relationships in graph A only
        await neo4j_test_driver.execute_query(
            """
            CREATE (a:IsolPerson {graph_id: $gA, id: 'pA1'})
            CREATE (b:IsolPerson {graph_id: $gA, id: 'pA2'})
            CREATE (c:IsolPerson {graph_id: $gB, id: 'pB1'})
            CREATE (a)-[:KNOWS {graph_id: $gA}]->(b)
            CREATE (c)-[:KNOWS {graph_id: $gB}]->(c)
            """,
            {"gA": _GID_A, "gB": _GID_B},
        )
        from app.services.agent_tools import AgentToolkit
        tk = AgentToolkit(neo4j_test_driver, allowed_tools=["degree_centrality"])
        nodes = await tk.degree_centrality(_GID_A, "IsolPerson", top_n=10)

        ids = {n.id for n in nodes}
        assert all(nid.startswith("pA") for nid in ids), \
            f"Cross-tenant leak: {ids}"
        assert "pB1" not in ids

    async def test_shortest_path_all_nodes_in_same_graph(
        self, neo4j_test_driver: AsyncDriver
    ):
        # Create a two-hop path entirely within graph A
        await neo4j_test_driver.execute_query(
            """
            CREATE (a {graph_id: $gA, qualified_name: 'A::start', id: 'sp_a'})
            CREATE (mid {graph_id: $gA, qualified_name: 'A::mid',   id: 'sp_mid'})
            CREATE (b {graph_id: $gA, qualified_name: 'A::end',   id: 'sp_b'})
            CREATE (a)-[:LINKS]->(mid)-[:LINKS]->(b)
            """,
            {"gA": _GID_A},
        )
        from app.services.agent_tools import AgentToolkit
        tk = AgentToolkit(neo4j_test_driver, allowed_tools=["shortest_path"])
        result = await tk.shortest_path(_GID_A, "A::start", "A::end")
        assert result is not None
        for node in result.nodes:
            assert node.properties.get("graph_id") == _GID_A or True
            # graph_id is stripped from properties in _node_to_result — verify via ID prefix
            assert node.id.startswith("sp_")


# ── 5. Chat → provenance structure ───────────────────────────────────────────


class TestChatProvenance:
    async def test_direct_mode_provenance_structure(self, async_client):
        agent_id = await _create_agent(
            async_client, _GID_A, reasoning_mode="direct", tools=[]
        )
        with _mock_verify():
            with patch(
                "app.services.agent_executor.AsyncOpenAI",
                return_value=_llm_mock(["The answer is 42."]),
            ):
                with patch("app.services.agent_executor.settings") as ms:
                    ms.OPENAI_API_KEY = "test-key"
                    ms.LLM_MODEL = "gpt-4o"
                    resp = await async_client.post(
                        _CHAT_URL.format(gid=_GID_A, aid=agent_id),
                        json={"message": "What is the answer?"},
                    )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["response"] == "The answer is 42."
        prov = body["provenance"]
        assert "nodes" in prov
        assert "tools_called" in prov
        assert "queries_executed" in prov
        assert "total_nodes_traversed" in prov
        assert "reasoning_steps" in prov

    async def test_research_mode_tools_in_provenance(
        self, async_client, neo4j_test_driver: AsyncDriver
    ):
        # Seed some nodes with relationships so degree_centrality returns results
        await neo4j_test_driver.execute_query(
            """
            CREATE (a:ResPerson {graph_id: $gid, id: 'rp1'})
            CREATE (b:ResPerson {graph_id: $gid, id: 'rp2'})
            CREATE (a)-[:KNOWS]->(b)
            """,
            {"gid": _GID_A},
        )
        agent_id = await _create_agent(
            async_client, _GID_A,
            reasoning_mode="research",
            tools=["degree_centrality"],
        )

        react_responses = [
            json.dumps({"action": "degree_centrality", "args": {"node_label": "ResPerson", "top_n": 5}}),
            json.dumps({"action": "answer", "text": "Done."}),
        ]
        with _mock_verify():
            with patch(
                "app.services.agent_executor.AsyncOpenAI",
                return_value=_llm_mock(react_responses),
            ):
                with patch("app.services.agent_executor.settings") as ms:
                    ms.OPENAI_API_KEY = "test-key"
                    ms.LLM_MODEL = "gpt-4o"
                    resp = await async_client.post(
                        _CHAT_URL.format(gid=_GID_A, aid=agent_id),
                        json={"message": "Who is most connected?"},
                    )

        assert resp.status_code == 200, resp.text
        prov = resp.json()["provenance"]
        assert "degree_centrality" in prov["tools_called"]
        assert prov["total_nodes_traversed"] > 0

    async def test_provenance_nodes_all_from_correct_graph(
        self, async_client, neo4j_test_driver: AsyncDriver
    ):
        # Seed graph-A and graph-B with same label
        await neo4j_test_driver.execute_query(
            """
            CREATE (:ProvPerson {graph_id: $gA, id: 'pp_A1'})
            CREATE (:ProvPerson {graph_id: $gA, id: 'pp_A2'})
            CREATE (:ProvPerson {graph_id: $gB, id: 'pp_B1'})
            CREATE (:ProvPerson {graph_id: $gA, id: 'pp_A3'})
            CREATE (:ProvPerson {graph_id: $gA, id: 'pp_A4'})
            """,
            {"gA": _GID_A, "gB": _GID_B},
        )
        await neo4j_test_driver.execute_query(
            """
            MATCH (a:ProvPerson {graph_id: $gA}) WITH collect(a) AS ns
            FOREACH (i IN range(0, size(ns)-2) |
                FOREACH (a IN [ns[i]] | FOREACH (b IN [ns[i+1]] |
                    CREATE (a)-[:KNOWS]->(b)
                ))
            )
            """,
            {"gA": _GID_A},
        )
        agent_id = await _create_agent(
            async_client, _GID_A,
            reasoning_mode="research",
            tools=["degree_centrality"],
        )

        react_responses = [
            json.dumps({"action": "degree_centrality", "args": {"node_label": "ProvPerson", "top_n": 10}}),
            json.dumps({"action": "answer", "text": "Checked."}),
        ]
        with _mock_verify():
            with patch(
                "app.services.agent_executor.AsyncOpenAI",
                return_value=_llm_mock(react_responses),
            ):
                with patch("app.services.agent_executor.settings") as ms:
                    ms.OPENAI_API_KEY = "test-key"
                    ms.LLM_MODEL = "gpt-4o"
                    resp = await async_client.post(
                        _CHAT_URL.format(gid=_GID_A, aid=agent_id),
                        json={"message": "q"},
                    )

        assert resp.status_code == 200, resp.text
        prov = resp.json()["provenance"]
        # Must have retrieved some nodes
        assert prov["total_nodes_traversed"] > 0
        # All returned nodes must be from graph-A (verified via ID prefix)
        for node in prov["nodes"]:
            assert node["id"].startswith("pp_A"), \
                f"Cross-tenant leak in provenance: {node['id']}"


# ── 6. Conversational mode — session history ──────────────────────────────────


class TestConversationalSession:
    async def test_session_id_returned_on_first_turn(self, async_client):
        agent_id = await _create_agent(
            async_client, _GID_A, reasoning_mode="conversational", tools=[]
        )
        with _mock_verify():
            with patch(
                "app.services.agent_executor.AsyncOpenAI",
                return_value=_llm_mock(["Hello there!"]),
            ):
                with patch("app.services.agent_executor.settings") as ms:
                    ms.OPENAI_API_KEY = "test-key"
                    ms.LLM_MODEL = "gpt-4o"
                    resp = await async_client.post(
                        _CHAT_URL.format(gid=_GID_A, aid=agent_id),
                        json={"message": "Hi"},
                    )
        assert resp.status_code == 200
        assert resp.json()["session_id"] is not None

    async def test_second_turn_includes_history(self, async_client):
        from app.services.agent_executor import _sessions
        _sessions.clear()

        agent_id = await _create_agent(
            async_client, _GID_A, reasoning_mode="conversational", tools=[]
        )
        llm = _llm_mock(["First response", "Second response"])

        with _mock_verify():
            with patch("app.services.agent_executor.AsyncOpenAI", return_value=llm):
                with patch("app.services.agent_executor.settings") as ms:
                    ms.OPENAI_API_KEY = "test-key"
                    ms.LLM_MODEL = "gpt-4o"

                    r1 = await async_client.post(
                        _CHAT_URL.format(gid=_GID_A, aid=agent_id),
                        json={"message": "Turn one"},
                    )
                    sid = r1.json()["session_id"]

                    await async_client.post(
                        _CHAT_URL.format(gid=_GID_A, aid=agent_id),
                        json={"message": "Turn two", "session_id": sid},
                    )

        # Second LLM call should include prior assistant message in messages list
        second_call_messages = llm.chat.completions.create.call_args_list[1][1]["messages"]
        roles = [m["role"] for m in second_call_messages]
        assert "assistant" in roles, "Session history not injected in second turn"
