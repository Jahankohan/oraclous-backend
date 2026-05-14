"""JSON Schemas for ``AgentToolkit`` tools, formatted per-provider.

These schemas drive the LLM's native tool-use protocol. Hand-written
rather than introspected from method signatures: hand-writing lets us
craft descriptions optimized for LLM tool-selection accuracy without
leaking implementation details (e.g. ``graph_id`` is injected by the
executor, never asked of the LLM).

Two output formats:

- ``openai`` — used by OpenRouter and Azure OpenAI clients (any
  ``AsyncOpenAI`` instance). Shape:
  ``{"type": "function", "function": {"name", "description", "parameters"}}``

- ``anthropic`` — used by the native ``AsyncAnthropic`` SDK. Shape:
  ``{"name", "description", "input_schema"}``

The parameter schemas are identical between formats; only the wrapper
differs. Adding a new tool: append one ``_ToolSchema`` entry below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ProviderFormat = Literal["openai", "anthropic"]


@dataclass(frozen=True, slots=True)
class _ToolSchema:
    """Canonical schema for one tool, format-agnostic."""

    name: str
    description: str
    parameters: dict[str, Any]


# JSON Schema fragment reused for ``max_results`` style integer caps.
def _int_param(
    description: str,
    *,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "type": "integer",
        "description": description,
        "default": default,
        "minimum": minimum,
    }
    if maximum is not None:
        spec["maximum"] = maximum
    return spec


# ── Tool schemas ─────────────────────────────────────────────────────────────
#
# ``graph_id`` is intentionally NOT a parameter on any tool. The executor
# binds it from the agent's configured ``graph_id`` before dispatch, so
# the LLM cannot accidentally read another tenant's data even if it
# hallucinated a different value.

_TOOL_SCHEMAS: dict[str, _ToolSchema] = {
    "graph_search": _ToolSchema(
        name="graph_search",
        description=(
            "Semantic similarity search over the knowledge graph's text "
            "embeddings. Returns the most relevant chunks/entities for a "
            "natural-language query. Use this when you need grounding "
            "context but don't know specific entity ids."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query to embed and match.",
                },
                "max_results": _int_param(
                    "How many top-scoring nodes to return.",
                    default=10,
                    minimum=1,
                    maximum=50,
                ),
            },
            "required": ["query"],
        },
    ),
    "community_members": _ToolSchema(
        name="community_members",
        description=(
            "Return the member nodes of a community, given a community_id. "
            "Works across every registered community kind (entity-Leiden "
            "and chunk-Louvain). Use this after find_communities or after "
            "spotting a community_id in another tool's output."
        ),
        parameters={
            "type": "object",
            "properties": {
                "community_id": {
                    "type": "string",
                    "description": "Stable id of the community to expand.",
                },
                "max_results": _int_param(
                    "How many members to return.",
                    default=50,
                    minimum=1,
                    maximum=200,
                ),
            },
            "required": ["community_id"],
        },
    ),
    "neighbors": _ToolSchema(
        name="neighbors",
        description=(
            "Breadth-first traversal from a known node, optionally "
            "filtered by relationship type. Use this for 'what's "
            "connected to X' and multi-hop reasoning (depth 2-3)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Source node id to traverse from.",
                },
                "edge_type": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional relationship type to filter on (e.g. "
                        "USES, OWNS). Omit to traverse all edges."
                    ),
                },
                "depth": _int_param(
                    "Hops from the source. Capped at 20 server-side.",
                    default=1,
                    minimum=1,
                    maximum=20,
                ),
            },
            "required": ["node_id"],
        },
    ),
    "degree_centrality": _ToolSchema(
        name="degree_centrality",
        description=(
            "Return the most-connected nodes of a given Neo4j label. "
            "Use this to find hub nodes / influential entities."
        ),
        parameters={
            "type": "object",
            "properties": {
                "node_label": {
                    "type": "string",
                    "description": (
                        "Neo4j label to rank (alphanumeric + underscore "
                        "only). Examples: Company, Person, __Entity__."
                    ),
                },
                "top_n": _int_param(
                    "How many top-ranked nodes to return.",
                    default=10,
                    minimum=1,
                    maximum=100,
                ),
            },
            "required": ["node_label"],
        },
    ),
    "shortest_path": _ToolSchema(
        name="shortest_path",
        description=(
            "Find the shortest path between two known nodes by "
            "qualified_name. Use this to explain how two concepts are "
            "related in the graph."
        ),
        parameters={
            "type": "object",
            "properties": {
                "from_qname": {
                    "type": "string",
                    "description": "qualified_name of the source node.",
                },
                "to_qname": {
                    "type": "string",
                    "description": "qualified_name of the target node.",
                },
            },
            "required": ["from_qname", "to_qname"],
        },
    ),
    "taint_trace": _ToolSchema(
        name="taint_trace",
        description=(
            "Follow FLOWS_TO edges from a source (code knowledge graphs "
            "only). Use this for security/data-flow questions: 'where "
            "does input X end up'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source_qname": {
                    "type": "string",
                    "description": "qualified_name of the taint source.",
                },
                "depth": _int_param(
                    "Maximum hops to follow. Capped at 20.",
                    default=10,
                    minimum=1,
                    maximum=20,
                ),
            },
            "required": ["source_qname"],
        },
    ),
    "cypher_query": _ToolSchema(
        name="cypher_query",
        description=(
            "Execute a read-only Cypher query against the graph. Use "
            "this for counts, aggregations, and structured lookups that "
            "graph_search can't answer.\n\n"
            "Hard requirements:\n"
            "- READ-ONLY: no CREATE / MERGE / DELETE / SET / REMOVE / "
            "DROP / DETACH DELETE.\n"
            "- MUST filter by graph_id for tenant isolation. Use the "
            "parameter $graph_id, e.g. "
            "'MATCH (n:__Entity__) WHERE n.graph_id = $graph_id RETURN "
            "count(n) AS total'.\n"
            "- LIMIT clause auto-injected if absent."
        ),
        parameters={
            "type": "object",
            "properties": {
                "cypher": {
                    "type": "string",
                    "description": (
                        "Read-only Cypher query. Must reference "
                        "$graph_id for tenant isolation."
                    ),
                },
                "max_results": _int_param(
                    "Cap on returned rows. Server enforces ≤ 100.",
                    default=25,
                    minimum=1,
                    maximum=100,
                ),
            },
            "required": ["cypher"],
        },
    ),
    "temporal_slice": _ToolSchema(
        name="temporal_slice",
        description=(
            "Return nodes that were valid at a given point in time "
            "(bitemporal valid_from / valid_to). Use this for "
            "'what did the graph look like on date X' questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "node_label": {
                    "type": "string",
                    "description": (
                        "Neo4j label to slice (alphanumeric + underscore)."
                    ),
                },
                "at_time": {
                    "type": "integer",
                    "description": (
                        "Unix epoch timestamp (seconds) of the point in time."
                    ),
                },
                "max_results": _int_param(
                    "Cap on returned nodes.",
                    default=50,
                    minimum=1,
                    maximum=200,
                ),
            },
            "required": ["node_label", "at_time"],
        },
    ),
}


def _to_openai(spec: _ToolSchema) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _to_anthropic(spec: _ToolSchema) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.parameters,
    }


def tool_schemas_for(
    allowed_tools: list[str] | set[str], provider_format: ProviderFormat
) -> list[dict[str, Any]]:
    """Return the per-provider tool schemas for the agent's allowlist.

    Unknown / unregistered tools in ``allowed_tools`` are silently
    dropped — the toolkit's ``ToolNotPermittedError`` still guards
    dispatch, so an unknown tool can't be called anyway. We don't fail
    here because the agent may legitimately have an allowlist that
    includes tools we haven't built JSON schemas for yet.
    """
    allowed = set(allowed_tools)
    selected = [s for name, s in _TOOL_SCHEMAS.items() if name in allowed]
    convert = _to_openai if provider_format == "openai" else _to_anthropic
    return [convert(s) for s in selected]


def registered_tool_names() -> list[str]:
    """Names of every tool with a JSON schema. Used by tests to assert
    every AgentToolkit method has a matching schema."""
    return list(_TOOL_SCHEMAS.keys())
