"""Unit tests for chat_engine — the mode→tool mapping + ChatResponse
adapter that unifies /chat with AgentExecutor (STORY-8)."""

import pytest

from app.services.chat_engine import (
    MODE_TO_TOOL,
    build_default_agent_config,
    derive_grounding,
    mode_to_retriever_label,
    tool_for_mode,
)


class TestToolForMode:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "mode,expected_tool",
        [
            ("simple", "graph_search"),
            ("enhanced", "vector_cypher_search"),
            ("hybrid", "hybrid_cypher_search"),
            ("hybrid_plus", "hybrid_cypher_search"),
            # ``natural`` routes to vector_cypher_search for STORY-8.
            # cypher_query expects an LLM-generated Cypher string and
            # can't be auto-dispatched in direct mode; threading text-
            # to-Cypher through research mode is a follow-up.
            ("natural", "vector_cypher_search"),
        ],
    )
    def test_known_modes_map_to_right_tools(self, mode, expected_tool):
        assert tool_for_mode(mode) == expected_tool

    @pytest.mark.unit
    def test_unknown_mode_falls_back_to_vector_cypher_search(self):
        """Unknown modes default to enhanced retrieval — the most useful
        fallback when a new mode is added without a matching tool."""
        assert tool_for_mode("brand_new_mode") == "vector_cypher_search"

    @pytest.mark.unit
    def test_none_mode_falls_back_to_default(self):
        assert tool_for_mode(None) == "vector_cypher_search"

    @pytest.mark.unit
    def test_case_insensitive(self):
        """Mode lookup should be case-insensitive — frontend can pass
        any casing."""
        assert tool_for_mode("ENHANCED") == "vector_cypher_search"

    @pytest.mark.unit
    def test_hybrid_and_hybrid_plus_alias_to_same_tool(self):
        """Per STORY-8 decision: both 'hybrid' and 'hybrid_plus' route to
        ``hybrid_cypher_search`` — graph capabilities on both."""
        assert tool_for_mode("hybrid") == tool_for_mode("hybrid_plus")


class TestBuildDefaultAgentConfig:
    @pytest.mark.unit
    def test_returns_dict_with_required_executor_keys(self):
        cfg = build_default_agent_config("g1", "enhanced")
        for key in (
            "agent_id",
            "graph_id",
            "name",
            "system_prompt",
            "reasoning_mode",
            "tools",
        ):
            assert key in cfg

    @pytest.mark.unit
    def test_graph_id_propagated(self):
        cfg = build_default_agent_config("my-graph-uuid", "simple")
        assert cfg["graph_id"] == "my-graph-uuid"

    @pytest.mark.unit
    def test_uses_direct_reasoning_mode(self):
        """STORY-8: chat is direct mode (one retrieval + one LLM call).
        Users wanting multi-step should hit /agents/{id}/chat with a
        research-mode agent."""
        cfg = build_default_agent_config("g1", "enhanced")
        assert cfg["reasoning_mode"] == "direct"

    @pytest.mark.unit
    def test_tools_contains_primary_tool_for_mode(self):
        cfg = build_default_agent_config("g1", "hybrid")
        assert "hybrid_cypher_search" in cfg["tools"]

    @pytest.mark.unit
    def test_extra_tools_appended(self):
        cfg = build_default_agent_config(
            "g1", "enhanced", extra_tools=["find_communities"]
        )
        assert "vector_cypher_search" in cfg["tools"]
        assert "find_communities" in cfg["tools"]

    @pytest.mark.unit
    def test_system_prompt_enforces_grounding(self):
        cfg = build_default_agent_config("g1", "enhanced")
        sp = cfg["system_prompt"].lower()
        # Two key guardrails — pinned so they can't be silently weakened
        assert "grounded" in sp or "ground" in sp
        assert "do not speculate" in sp or "do not invent" in sp

    @pytest.mark.unit
    def test_synthetic_agent_id_is_constant(self):
        """The synthetic agent is never persisted; its id is a marker
        so log filtering by ``agent_id != 'chat-default'`` works."""
        cfg = build_default_agent_config("g1", "enhanced")
        assert cfg["agent_id"] == "chat-default"


class TestDeriveGrounding:
    @pytest.mark.unit
    def test_empty_nodes_means_not_grounded(self):
        is_grounded, confidence = derive_grounding("any text", [])
        assert is_grounded is False
        assert confidence == 0.0

    @pytest.mark.unit
    def test_nodes_present_but_uncited_yields_base_confidence(self):
        """When the retriever returned nodes but the LLM didn't cite
        any of them, we're grounded (retrieval happened) but
        confidence is a low base value."""
        nodes = [{"id": "n1", "label": "x"}, {"id": "n2", "label": "y"}]
        is_grounded, confidence = derive_grounding("totally unrelated answer", nodes)
        assert is_grounded is True
        assert 0.0 < confidence < 1.0

    @pytest.mark.unit
    def test_three_cited_nodes_saturates_confidence(self):
        nodes = [{"id": f"n{i}", "label": "x"} for i in range(5)]
        # Response cites three of them by id
        response = "Answer references n1, n2, and n3 explicitly."
        is_grounded, confidence = derive_grounding(response, nodes)
        assert is_grounded is True
        assert confidence == 1.0

    @pytest.mark.unit
    def test_one_cited_node_yields_one_third_confidence(self):
        nodes = [{"id": f"n{i}", "label": "x"} for i in range(5)]
        response = "Answer references n0 only."
        is_grounded, confidence = derive_grounding(response, nodes)
        # 1 cited / 3.0 = 0.333
        assert 0.3 < confidence < 0.4

    @pytest.mark.unit
    def test_nodes_without_ids_dont_break(self):
        """Some provenance entries may have None for id — derive
        shouldn't crash on missing keys."""
        nodes = [{"id": None, "label": "x"}, {"id": "n1", "label": "y"}]
        is_grounded, _confidence = derive_grounding("mentions n1", nodes)
        assert is_grounded is True


class TestRetrieverLabel:
    @pytest.mark.unit
    def test_label_matches_tool_name(self):
        """The response field tells the caller which tool backed the
        retrieval — useful for debugging frontend behaviour."""
        assert mode_to_retriever_label("enhanced") == "vector_cypher_search"
        assert mode_to_retriever_label("hybrid") == "hybrid_cypher_search"


class TestModeMapStability:
    @pytest.mark.unit
    def test_mode_map_covers_chat_modes(self):
        """The constant MODE_TO_TOOL must cover every value of ChatMode.
        Pin this so adding a new mode without a tool gets a clear test
        failure rather than a silent fallback at runtime."""
        from app.schemas.chat_schemas import ChatMode

        for mode in ChatMode:
            assert mode.value in MODE_TO_TOOL, (
                f"ChatMode.{mode.name} ({mode.value!r}) has no entry in "
                "MODE_TO_TOOL — add it before the runtime fallback masks it."
            )
