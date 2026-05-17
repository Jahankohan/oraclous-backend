"""Unit tests for AgentExecutor — four reasoning modes and provenance (TASK-034 / STORY-020).

Every test mocks:
- The AgentToolkit (no Neo4j I/O)
- The AsyncOpenAI client (no real LLM calls)
Tests verify reasoning mode behaviour, provenance collection, session handling,
and error surfacing.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.agent_schemas import AgentChatResponse, NodeResult
from app.services.agent_executor import AgentExecutor, _sessions
from app.services.agent_tools import AgentToolkit, ToolNotPermittedError
from app.services.provenance import ProvenanceCollector

# ── Test helpers ──────────────────────────────────────────────────────────────


def _make_agent(
    graph_id: str = "g1",
    mode: str = "direct",
    tools: list[str] | None = None,
    system_prompt: str = "You are helpful.",
    llm_config_id: str | None = None,
) -> dict:
    return {
        "agent_id": "agent-1",
        "graph_id": graph_id,
        "name": "Test",
        "system_prompt": system_prompt,
        "reasoning_mode": mode,
        "tools": tools if tools is not None else ["graph_search"],
        "llm_config_id": llm_config_id,
    }


def _make_toolkit(graph_search_returns=None) -> AgentToolkit:
    nodes = graph_search_returns or [NodeResult(id="n1", label="Node", properties={})]
    tk = MagicMock(spec=AgentToolkit)
    tk.graph_search = AsyncMock(return_value=nodes)
    tk.community_members = AsyncMock(return_value=[])
    tk.neighbors = AsyncMock(return_value=[])
    tk.degree_centrality = AsyncMock(return_value=[])
    tk.shortest_path = AsyncMock(return_value=None)
    tk.taint_trace = AsyncMock(return_value=[])
    tk.temporal_slice = AsyncMock(return_value=[])
    return tk


def _make_llm(responses: list[str]) -> MagicMock:
    """Build a mock AsyncOpenAI client returning responses in order.

    Each ``responses`` entry is a plain string with no tool_calls.
    For tool-use scenarios use ``_make_llm_with_tool_calls`` instead.
    """
    client = MagicMock()
    completions = []
    for text in responses:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = text
        mock_resp.choices[0].message.tool_calls = None
        completions.append(mock_resp)
    client.chat.completions.create = AsyncMock(side_effect=completions)
    return client


def _tool_call(call_id: str, name: str, args: dict) -> MagicMock:
    """Build a mock OpenAI-shape tool_call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _make_llm_with_tool_calls(turns: list[dict]) -> MagicMock:
    """Build a mock AsyncOpenAI client where each turn either invokes
    tools or returns final text.

    Each ``turn`` is one of:
      - {"text": "...final answer..."}
      - {"tool_calls": [(call_id, name, args), ...], "text": ""}
    """
    client = MagicMock()
    completions = []
    for turn in turns:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = turn.get("text", "") or ""
        raw_tcs = turn.get("tool_calls") or []
        mock_resp.choices[0].message.tool_calls = (
            [_tool_call(*t) for t in raw_tcs] if raw_tcs else None
        )
        completions.append(mock_resp)
    client.chat.completions.create = AsyncMock(side_effect=completions)
    return client


def _executor(agent=None, toolkit=None, llm=None, responses=None) -> AgentExecutor:
    if agent is None:
        agent = _make_agent()
    if toolkit is None:
        toolkit = _make_toolkit()
    if llm is None:
        llm = _make_llm(responses or ["Default response."])
    return AgentExecutor(agent, toolkit, llm, model="gpt-4o")


# ── ProvenanceCollector ───────────────────────────────────────────────────────


class TestProvenanceCollector:
    def test_empty_payload(self):
        prov = ProvenanceCollector()
        p = prov.to_payload()
        assert p.total_nodes_traversed == 0
        assert p.reasoning_steps == 0
        assert p.tools_called == []

    def test_record_tool_adds_nodes(self):
        prov = ProvenanceCollector()
        prov.record_tool(
            "graph_search", [NodeResult(id="n1", label="X", properties={})]
        )
        p = prov.to_payload()
        assert p.total_nodes_traversed == 1
        assert p.reasoning_steps == 1
        assert "graph_search" in p.tools_called

    def test_record_multiple_tools(self):
        prov = ProvenanceCollector()
        prov.record_tool(
            "graph_search", [NodeResult(id="n1", label="A", properties={})]
        )
        prov.record_tool("neighbors", [NodeResult(id="n2", label="B", properties={})])
        p = prov.to_payload()
        assert p.total_nodes_traversed == 2
        assert p.reasoning_steps == 2

    def test_nodes_used_in_response_settable(self):
        prov = ProvenanceCollector()
        prov.nodes_used_in_response = ["n1", "n2"]
        p = prov.to_payload()
        assert p.nodes_used_in_response == ["n1", "n2"]


# ── Direct mode ───────────────────────────────────────────────────────────────


class TestDirectMode:
    async def test_returns_agent_chat_response(self):
        ex = _executor(responses=["The answer."])
        result = await ex.run("What is X?", session_id=None)
        assert isinstance(result, AgentChatResponse)
        assert result.response == "The answer."

    async def test_calls_graph_search_once(self):
        tk = _make_toolkit()
        ex = _executor(toolkit=tk, responses=["ok"])
        await ex.run("question", session_id=None)
        tk.graph_search.assert_called_once()

    async def test_graph_id_passed_to_graph_search(self):
        # TASK-205: _dispatch passes the effective graph-id *set* to the
        # tool. With no requesting user / no driver the set falls back to
        # exactly the source graph — a one-element list. The toolkit
        # treats `['my-graph']` identically to the legacy `'my-graph'`.
        tk = _make_toolkit()
        ex = _executor(
            agent=_make_agent(graph_id="my-graph"), toolkit=tk, responses=["ok"]
        )
        await ex.run("q", session_id=None)
        call_args = tk.graph_search.call_args
        assert call_args[0][0] == ["my-graph"]

    async def test_provenance_records_tool_call(self):
        ex = _executor(responses=["answer"])
        result = await ex.run("q", session_id=None)
        assert "graph_search" in result.provenance.tools_called

    async def test_provenance_nodes_populated(self):
        nodes = [NodeResult(id="abc", label="Thing", properties={})]
        tk = _make_toolkit(graph_search_returns=nodes)
        ex = _executor(toolkit=tk, responses=["the answer"])
        result = await ex.run("q", session_id=None)
        assert result.provenance.total_nodes_traversed == 1

    async def test_skips_graph_search_when_not_in_tools(self):
        tk = _make_toolkit()
        agent = _make_agent(tools=["neighbors"])
        ex = _executor(agent=agent, toolkit=tk, responses=["ok"])
        await ex.run("q", session_id=None)
        tk.graph_search.assert_not_called()

    async def test_at_most_two_llm_calls(self):
        llm = _make_llm(["resp1", "resp2", "resp3"])
        ex = _executor(llm=llm)
        await ex.run("q", session_id=None)
        assert llm.chat.completions.create.call_count <= 2

    async def test_nodes_used_in_response_contains_cited_id(self):
        nodes = [NodeResult(id="node-xyz", label="Thing", properties={})]
        tk = _make_toolkit(graph_search_returns=nodes)
        ex = _executor(toolkit=tk, responses=["The node-xyz is relevant."])
        result = await ex.run("q", session_id=None)
        assert "node-xyz" in result.provenance.nodes_used_in_response


# ── Research mode (native tool-use loop, STORY-5) ─────────────────────────────


class TestResearchMode:
    async def test_executes_at_least_two_tool_calls(self):
        tk = _make_toolkit()
        llm = _make_llm_with_tool_calls(
            [
                {"tool_calls": [("c1", "graph_search", {"query": "step1"})]},
                {"tool_calls": [("c2", "graph_search", {"query": "step2"})]},
                {"text": "Final answer"},
            ]
        )
        agent = _make_agent(mode="research", tools=["graph_search"])
        ex = _executor(agent=agent, toolkit=tk, llm=llm)
        await ex.run("What is Y?", session_id=None)
        assert tk.graph_search.call_count >= 2

    async def test_returns_final_text_when_no_tool_calls(self):
        tk = _make_toolkit()
        llm = _make_llm_with_tool_calls(
            [
                {"tool_calls": [("c1", "graph_search", {"query": "info"})]},
                {"text": "The definitive answer."},
            ]
        )
        agent = _make_agent(mode="research", tools=["graph_search"])
        ex = _executor(agent=agent, toolkit=tk, llm=llm)
        result = await ex.run("question", session_id=None)
        assert result.response == "The definitive answer."

    async def test_caps_at_five_iterations(self):
        tk = _make_toolkit()
        # Always emit a tool call — never a no-tool-call turn. Should cap at 5.
        looping = [
            {"tool_calls": [(f"c{i}", "graph_search", {"query": "x"})]}
            for i in range(10)
        ]
        agent = _make_agent(mode="research", tools=["graph_search"])
        ex = _executor(agent=agent, toolkit=tk, llm=_make_llm_with_tool_calls(looping))
        await ex.run("q", session_id=None)
        # _MAX_TOOL_ITERATIONS is 5
        assert tk.graph_search.call_count <= 5

    async def test_provenance_records_all_tool_calls(self):
        tk = _make_toolkit()
        llm = _make_llm_with_tool_calls(
            [
                {"tool_calls": [("c1", "graph_search", {"query": "a"})]},
                {"tool_calls": [("c2", "graph_search", {"query": "b"})]},
                {"text": "Done"},
            ]
        )
        agent = _make_agent(mode="research", tools=["graph_search"])
        ex = _executor(agent=agent, toolkit=tk, llm=llm)
        result = await ex.run("q", session_id=None)
        assert result.provenance.tools_called.count("graph_search") == 2

    async def test_provenance_records_tool_call_id(self):
        """Each tool_call entry should carry the LLM-provided id."""
        tk = _make_toolkit()
        llm = _make_llm_with_tool_calls(
            [
                {"tool_calls": [("call_abc", "graph_search", {"query": "x"})]},
                {"text": "Done"},
            ]
        )
        agent = _make_agent(mode="research", tools=["graph_search"])
        ex = _executor(agent=agent, toolkit=tk, llm=llm)
        result = await ex.run("q", session_id=None)
        ids = [tc["id"] for tc in result.provenance.tool_calls]
        assert "call_abc" in ids

    async def test_tool_error_continues_loop(self):
        """Tool exception is fed back as a tool-result error; loop continues."""
        tk = _make_toolkit()
        tk.graph_search = AsyncMock(side_effect=ToolNotPermittedError("graph_search"))
        llm = _make_llm_with_tool_calls(
            [
                {"tool_calls": [("c1", "graph_search", {"query": "x"})]},
                {"text": "Recovered answer"},
            ]
        )
        agent = _make_agent(mode="research", tools=[])
        ex = _executor(agent=agent, toolkit=tk, llm=llm)
        result = await ex.run("q", session_id=None)
        assert result.response == "Recovered answer"

    async def test_no_tool_calls_returns_text_immediately(self):
        """First-turn text answer (no tool_calls) returns directly."""
        tk = _make_toolkit()
        llm = _make_llm_with_tool_calls([{"text": "Immediate answer"}])
        agent = _make_agent(mode="research", tools=["graph_search"])
        ex = _executor(agent=agent, toolkit=tk, llm=llm)
        result = await ex.run("q", session_id=None)
        assert result.response == "Immediate answer"


# ── Analytical mode (now a tool-use loop too, STORY-5) ────────────────────────


class TestAnalyticalMode:
    async def test_returns_response(self):
        agent = _make_agent(mode="analytical")
        ex = _executor(
            agent=agent, responses=["Step 1: examined. Step 2: weighed. Conclusion: X"]
        )
        result = await ex.run("Analyse this.", session_id=None)
        assert result.response

    async def test_prompt_contains_structured_reasoning_instruction(self):
        """System prompt should nudge structured reasoning."""
        tk = _make_toolkit()
        llm = _make_llm(["Step 1: ... Conclusion: y"])
        agent = _make_agent(mode="analytical")
        ex = _executor(agent=agent, toolkit=tk, llm=llm)
        await ex.run("q", session_id=None)
        # System message is the first message in OpenAI-shape calls
        system_msg = llm.chat.completions.create.call_args[1]["messages"][0]["content"]
        assert "step by step" in system_msg.lower()

    async def test_does_not_force_graph_search(self):
        """Unlike pre-STORY-5 analytical, the loop only invokes tools the
        model chooses. With no tool_calls in the response, graph_search
        is NOT called automatically."""
        tk = _make_toolkit()
        agent = _make_agent(mode="analytical", tools=["graph_search"])
        ex = _executor(agent=agent, toolkit=tk, responses=["analysis result"])
        await ex.run("q", session_id=None)
        tk.graph_search.assert_not_called()

    async def test_dispatches_tool_when_model_calls_it(self):
        tk = _make_toolkit()
        llm = _make_llm_with_tool_calls(
            [
                {"tool_calls": [("c1", "graph_search", {"query": "x"})]},
                {"text": "Analysis complete"},
            ]
        )
        agent = _make_agent(mode="analytical", tools=["graph_search"])
        ex = _executor(agent=agent, toolkit=tk, llm=llm)
        result = await ex.run("q", session_id=None)
        assert tk.graph_search.call_count == 1
        assert "graph_search" in result.provenance.tools_called


# ── Conversational mode ───────────────────────────────────────────────────────


class TestConversationalMode:
    def setup_method(self):
        _sessions.clear()

    async def test_new_session_id_assigned(self):
        agent = _make_agent(mode="conversational")
        ex = _executor(agent=agent, responses=["Hello!"])
        result = await ex.run("Hi", session_id=None)
        assert result.session_id is not None

    async def test_same_session_id_returned(self):
        agent = _make_agent(mode="conversational")
        ex = _executor(agent=agent, responses=["Hello!", "Follow-up"])
        r1 = await ex.run("Turn 1", session_id=None)
        r2 = await ex.run("Turn 2", session_id=r1.session_id)
        assert r2.session_id == r1.session_id

    async def test_history_injected_in_subsequent_turns(self):
        tk = _make_toolkit()
        llm = _make_llm(["First response", "Second response"])
        agent = _make_agent(mode="conversational")
        ex = _executor(agent=agent, toolkit=tk, llm=llm)

        r1 = await ex.run("Turn 1", session_id=None)
        await ex.run("Turn 2", session_id=r1.session_id)

        # Second call should include history (prior user + assistant messages)
        second_call_messages = llm.chat.completions.create.call_args_list[1][1][
            "messages"
        ]
        roles = [m["role"] for m in second_call_messages]
        assert "assistant" in roles  # prior assistant turn injected

    async def test_system_prompt_has_conversational_suffix(self):
        tk = _make_toolkit()
        llm = _make_llm(["response"])
        agent = _make_agent(mode="conversational", system_prompt="Base prompt.")
        ex = _executor(agent=agent, toolkit=tk, llm=llm)
        await ex.run("q", session_id=None)
        system_msg = llm.chat.completions.create.call_args[1]["messages"][0]["content"]
        assert "ongoing conversation" in system_msg.lower()

    async def test_different_sessions_are_independent(self):
        agent = _make_agent(mode="conversational")
        tk = _make_toolkit()
        llm = _make_llm(["A1", "B1"])
        ex = _executor(agent=agent, toolkit=tk, llm=llm)

        r_a = await ex.run("Session A", session_id=None)
        r_b = await ex.run("Session B", session_id=None)
        assert r_a.session_id != r_b.session_id


# ── AgentExecutor.from_neo4j error paths ──────────────────────────────────────


class TestFromNeo4jErrors:
    async def test_missing_agent_raises_value_error(self):
        driver = MagicMock()
        with patch("app.services.agent_executor.AgentService") as MockSvc:
            MockSvc.return_value.get_agent = AsyncMock(return_value=None)
            with pytest.raises(ValueError, match="not found"):
                await AgentExecutor.from_neo4j(driver, "g", "missing-id")

    async def test_llm_config_id_resolved_via_chain(self):
        driver = MagicMock()
        agent = _make_agent(llm_config_id="some-config")
        resolved = {
            "config_id": "some-config",
            "provider": "openrouter",
            "model": "openai/gpt-4o",
            "base_url": "https://openrouter.ai/api/v1",
            "api_version": None,
            "api_key_ref": "cred-123",
            "deactivated_at": None,
        }
        with (
            patch("app.services.agent_executor.AgentService") as MockSvc,
            patch("app.services.agent_executor.LLMConfigService") as MockCfg,
            patch("app.services.agent_executor.CredentialBrokerClient") as MockBroker,
            patch("app.services.agent_executor.LLMClientFactory") as MockFactory,
        ):
            MockSvc.return_value.get_agent = AsyncMock(return_value=agent)
            MockCfg.return_value.resolve_for_agent = AsyncMock(return_value=resolved)
            MockBroker.return_value.retrieve_api_key = AsyncMock(
                return_value="sk-or-test"
            )
            MockFactory.build = MagicMock(return_value=MagicMock())
            executor = await AgentExecutor.from_neo4j(driver, "g", "a")
        assert executor is not None

    async def test_no_api_key_raises_runtime_error(self):
        driver = MagicMock()
        agent = _make_agent()
        with (
            patch("app.services.agent_executor.AgentService") as MockSvc,
            patch("app.services.agent_executor.LLMConfigService") as MockCfg,
        ):
            MockSvc.return_value.get_agent = AsyncMock(return_value=agent)
            MockCfg.return_value.resolve_for_agent = AsyncMock(return_value=None)
            with patch("app.services.agent_executor.settings") as mock_settings:
                mock_settings.LLM_API_KEY = None
                mock_settings.OPENAI_API_KEY = None
                mock_settings.LLM_MODEL = "gpt-4o"
                mock_settings.CREDENTIAL_BROKER_URL = "http://broker:8000"
                with pytest.raises(RuntimeError, match="LLM not configured"):
                    await AgentExecutor.from_neo4j(driver, "g", "a")
