"""Chat engine — unifies /chat with the AgentExecutor (STORY-8).

Pre-STORY-8 there were two separate engines:

- ``ChatService`` (1167 LOC) — neo4j_graphrag GraphRAG class, 5 retriever
  types, Redis cache, source-first SSE streaming.
- ``AgentExecutor`` (696 LOC, post-STORY-5) — native LLM tool-use, 4
  reasoning modes, provenance tracking.

After STORY-8 there's one: ``/chat`` builds a synthetic in-memory agent
config (no Neo4j persist), maps the user-facing ``mode`` to one of the
agent toolkit's retrieval tools, and routes through ``AgentExecutor.run``.
The streaming variant emits per-source SSE events first (preserving the
existing frontend contract) and then streams the LLM answer.

This module is the thin adapter layer. It owns:
- Mode → tool mapping
- Synthetic default-agent construction
- ChatResponse derivation from AgentChatResponse + AgentToolkit retrieval

The historical ``ChatService`` remains importable but is marked
deprecated; existing callers that haven't migrated still work. Modes
this module doesn't map yet (e.g. text2cypher) gracefully fall back to
the default ``vector_cypher_search`` tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.schemas.chat_schemas import ChatMode

logger = get_logger(__name__)


# The strict-grounding system prompt for chat. Pulled from ChatService's
# STRICT_GROUNDING_PROMPT so behaviour is preserved exactly when migrating.
# (When STORY-8 lands, this becomes the single source of truth.)
_CHAT_SYSTEM_PROMPT = """\
You are a strictly grounded knowledge-graph assistant. You answer ONLY
from the retrieved graph context shown to you via tool results. If the
context does not contain enough information to answer the user's
question, you say so explicitly and do not speculate.

Hard rules:
- Quote facts directly from the tool result content where possible.
- When you cite a chunk or entity, mention its identifier so the user
  can trace the source.
- If asked something the graph does not cover, respond with: "The
  knowledge graph does not contain sufficient data to answer this
  question." and explain what specifically is missing.
- Do not invent entities, relationships, dates, names, or properties
  that are not in the retrieved context.
"""


# Mode → AgentToolkit tool name. The user-facing modes from
# chat_schemas.ChatMode map to the tools added by STORY-8 and earlier.
# Falling back to vector_cypher_search for unknown / unmapped modes
# keeps behavior reasonable when new modes are added without a
# matching tool.
_MODE_TO_TOOL: dict[str, str] = {
    # Vector only (fastest, no graph traversal)
    "simple": "graph_search",
    # Vector + graph traversal — default
    "enhanced": "vector_cypher_search",
    # Hybrid (vector + fulltext BM25) + graph traversal
    "hybrid": "hybrid_cypher_search",
    # Alias of hybrid post-STORY-8 — both have "advanced graph capabilities"
    "hybrid_plus": "hybrid_cypher_search",
    # Text-to-Cypher. Routes to vector_cypher_search for STORY-8 because
    # cypher_query expects an LLM-generated Cypher string and can't be
    # auto-dispatched in direct mode. A follow-up will route mode=natural
    # through research mode so the LLM generates the Cypher itself.
    "natural": "vector_cypher_search",
}


def tool_for_mode(mode: str | None) -> str:
    """Map a user-facing chat mode to the AgentToolkit tool that backs
    it. Unknown modes fall through to ``vector_cypher_search``."""
    if not mode:
        return "vector_cypher_search"
    return _MODE_TO_TOOL.get(str(mode).lower(), "vector_cypher_search")


def build_default_agent_config(
    graph_id: str,
    mode: str | None,
    *,
    extra_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Build an ephemeral agent config for routing /chat through
    AgentExecutor.

    The result is never persisted to Neo4j — it's purely an in-memory
    description that AgentExecutor.__init__ accepts.

    Parameters
    ----------
    graph_id:
        Tenant graph (bound on the agent so the executor injects it
        into every tool call).
    mode:
        User-facing chat mode (simple / enhanced / hybrid / hybrid_plus
        / natural). Maps to one primary retrieval tool.
    extra_tools:
        Additional tools to expose. Today unused; future hook for chat
        configurations that want the LLM to also call e.g.
        find_communities mid-answer.

    Returns
    -------
    dict
        Shape compatible with ``AgentExecutor.__init__`` — same keys
        as a persisted :Agent node (agent_id, graph_id, name,
        system_prompt, reasoning_mode, tools).
    """
    primary_tool = tool_for_mode(mode)
    tools = [primary_tool] + list(extra_tools or [])
    return {
        "agent_id": "chat-default",
        "graph_id": graph_id,
        "name": "chat-default",
        "system_prompt": _CHAT_SYSTEM_PROMPT,
        # Direct mode = one retrieval + one LLM call. Matches the
        # pre-STORY-8 ChatService behaviour. Users wanting multi-step
        # reasoning should hit /agents/{id}/chat with a research-mode
        # agent.
        "reasoning_mode": "direct",
        "tools": tools,
    }


def derive_grounding(
    response_text: str,
    nodes: list[dict[str, Any]],
) -> tuple[bool, float]:
    """Derive (is_grounded, confidence) from provenance for the chat
    response shape.

    Pre-STORY-8 ChatService computed these from retriever scores and a
    set of heuristics; STORY-8 keeps the signal but simplifies the
    derivation to two facts that survive across all retrieval paths:

    - ``is_grounded`` ← whether the agent dispatched at least one
      retrieval tool that returned non-empty results.
    - ``confidence`` ← rough scale 0..1 based on how many retrieved
      nodes the final response actually cites (their id appears
      literally in the text). Three or more cited nodes saturates at 1.0.
    """
    if not nodes:
        return False, 0.0
    # Count how many of the retrieved nodes actually got cited in the
    # final answer. Heuristic — better than nothing, replaces the
    # retriever-score heuristic ChatService used.
    cited = sum(1 for n in nodes if n.get("id") and n.get("id") in response_text)
    confidence = min(1.0, cited / 3.0) if cited > 0 else 0.3
    return True, round(confidence, 3)


def mode_to_retriever_label(mode: ChatMode | str | None) -> str:
    """Best-effort string label for ``ChatResponse.retriever_type``
    (the response field that historically named the neo4j_graphrag
    retriever type). After STORY-8 the field labels the agent tool
    that backed the retrieval — keeps the contract informative for
    callers while reflecting the new engine."""
    return tool_for_mode(mode if mode else None)


# Public constants for callers that want to inspect the wiring without
# importing private names.
CHAT_SYSTEM_PROMPT = _CHAT_SYSTEM_PROMPT
MODE_TO_TOOL = dict(_MODE_TO_TOOL)
