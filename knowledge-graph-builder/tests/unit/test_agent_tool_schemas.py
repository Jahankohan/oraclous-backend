"""Unit tests for app/services/agent_tool_schemas.py (STORY-5).

These pin the tool-use contract: every AgentToolkit method that the
executor can dispatch has a matching JSON schema, schemas convert to
both provider formats without losing parameters, and graph_id is never
exposed to the LLM.
"""

import inspect

import pytest

from app.services.agent_tool_schemas import (
    _TOOL_SCHEMAS,
    registered_tool_names,
    tool_schemas_for,
)
from app.services.agent_tools import AgentToolkit


class TestRegistrationCoverage:
    @pytest.mark.unit
    def test_every_agent_toolkit_method_has_a_schema(self):
        """If an AgentToolkit public async method is dispatchable by the
        executor, it must have a JSON schema. New tools without schemas
        would silently be invisible to the LLM."""
        dispatchable = {
            name
            for name, member in inspect.getmembers(
                AgentToolkit, predicate=inspect.iscoroutinefunction
            )
            if not name.startswith("_")
        }
        registered = set(registered_tool_names())
        missing = dispatchable - registered
        assert not missing, (
            f"AgentToolkit methods without JSON schemas: {sorted(missing)}"
        )

    @pytest.mark.unit
    def test_no_orphan_schemas(self):
        """A registered schema must point at a real AgentToolkit method."""
        toolkit_methods = {
            name
            for name, member in inspect.getmembers(
                AgentToolkit, predicate=inspect.iscoroutinefunction
            )
        }
        for name in registered_tool_names():
            assert name in toolkit_methods, (
                f"Schema {name!r} doesn't match any AgentToolkit method"
            )


class TestGraphIdNeverInSchema:
    @pytest.mark.unit
    @pytest.mark.parametrize("name", registered_tool_names())
    def test_graph_id_is_not_a_parameter(self, name):
        """``graph_id`` is bound by the executor before dispatch — the LLM
        must never see it as a parameter, even hallucinating one shouldn't
        be possible. Pin this hard."""
        spec = _TOOL_SCHEMAS[name]
        params = spec.parameters
        assert "graph_id" not in params.get("properties", {}), (
            f"Tool {name!r} exposes graph_id to the LLM — tenant-isolation risk"
        )
        assert "graph_id" not in params.get("required", []), (
            f"Tool {name!r} requires graph_id from the LLM — tenant-isolation risk"
        )


class TestProviderFormats:
    @pytest.mark.unit
    def test_openai_format_shape(self):
        out = tool_schemas_for(["graph_search"], "openai")
        assert len(out) == 1
        s = out[0]
        assert s["type"] == "function"
        assert s["function"]["name"] == "graph_search"
        assert "parameters" in s["function"]
        assert "description" in s["function"]

    @pytest.mark.unit
    def test_anthropic_format_shape(self):
        out = tool_schemas_for(["graph_search"], "anthropic")
        assert len(out) == 1
        s = out[0]
        assert s["name"] == "graph_search"
        assert "input_schema" in s
        assert "description" in s
        # Anthropic format does NOT wrap in "function"
        assert "function" not in s
        assert "type" not in s or s.get("type") != "function"

    @pytest.mark.unit
    def test_filtering_by_allowlist(self):
        """Only the tools in the allowlist are returned."""
        out_names = [
            s["function"]["name"]
            for s in tool_schemas_for(["graph_search", "neighbors"], "openai")
        ]
        assert set(out_names) == {"graph_search", "neighbors"}

    @pytest.mark.unit
    def test_unknown_tool_silently_dropped(self):
        """Unknown tools in the allowlist don't crash schema construction —
        ToolNotPermittedError at dispatch time covers them."""
        out = tool_schemas_for(["graph_search", "nonexistent_tool"], "openai")
        names = [s["function"]["name"] for s in out]
        assert "graph_search" in names
        assert "nonexistent_tool" not in names

    @pytest.mark.unit
    def test_empty_allowlist_returns_empty(self):
        assert tool_schemas_for([], "openai") == []
        assert tool_schemas_for([], "anthropic") == []


class TestParameterShapes:
    @pytest.mark.unit
    def test_cypher_query_requires_cypher(self):
        spec = _TOOL_SCHEMAS["cypher_query"]
        assert "cypher" in spec.parameters["required"]
        # Must describe the read-only + graph_id constraints so the LLM
        # writes correct Cypher on its first try
        assert "graph_id" in spec.description.lower()
        assert "read-only" in spec.description.lower()

    @pytest.mark.unit
    def test_max_results_caps_match_server_side(self):
        """JSON Schema max should mirror the server-side cap so the model
        doesn't burn turns asking for more than will be returned."""
        # cypher_query is capped at 100 in agent_tools.py
        cypher_params = _TOOL_SCHEMAS["cypher_query"].parameters["properties"]
        assert cypher_params["max_results"]["maximum"] == 100
        # graph_search is unbounded server-side; schema caps at 50 conservatively
        gs_params = _TOOL_SCHEMAS["graph_search"].parameters["properties"]
        assert gs_params["max_results"]["maximum"] == 50
