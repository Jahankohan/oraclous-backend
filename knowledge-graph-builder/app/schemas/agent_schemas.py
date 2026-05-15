"""Pydantic schemas for Graph-Native Agent CRUD and execution."""

from typing import Literal

from pydantic import BaseModel, field_validator


# Valid tool names — enforced at creation time so no invalid tool can be
# persisted. Sourced from the agent_tool_schemas registry so adding a new
# tool is one-touch (add a schema entry; the validator picks it up
# automatically and the LLM-facing JSON schemas + this server-side
# validator never drift apart).
def _load_valid_tools() -> frozenset[str]:
    from app.services.agent_tool_schemas import registered_tool_names

    return frozenset(registered_tool_names())


VALID_TOOLS = _load_valid_tools()

ReasoningMode = Literal["direct", "research", "analytical", "conversational"]
RetrieverStrategy = Literal["hybrid", "similarity", "entity"]


class RetrieverConfig(BaseModel):
    strategy: RetrieverStrategy = "hybrid"
    hop_depth: int = 2
    max_results: int = 20


class AgentCreate(BaseModel):
    name: str
    description: str = ""
    system_prompt: str
    reasoning_mode: ReasoningMode = "direct"
    retriever: RetrieverConfig = RetrieverConfig()
    tools: list[str] = ["graph_search"]
    llm_config_id: str | None = None

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[str]) -> list[str]:
        unknown = set(v) - VALID_TOOLS
        if unknown:
            raise ValueError(
                f"Unknown tool(s): {sorted(unknown)}. Valid tools: {sorted(VALID_TOOLS)}"
            )
        return v


class AgentUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    reasoning_mode: ReasoningMode | None = None
    retriever: RetrieverConfig | None = None
    tools: list[str] | None = None
    llm_config_id: str | None = None

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        unknown = set(v) - VALID_TOOLS
        if unknown:
            raise ValueError(
                f"Unknown tool(s): {sorted(unknown)}. Valid tools: {sorted(VALID_TOOLS)}"
            )
        return v


class AgentResponse(BaseModel):
    agent_id: str
    graph_id: str
    name: str
    description: str
    system_prompt: str
    reasoning_mode: str
    retriever: RetrieverConfig
    tools: list[str]
    llm_config_id: str | None
    created_by: str
    created_at: str
    deactivated_at: str | None


class AgentCreateResponse(BaseModel):
    agent_id: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    # Persisted-conversation id (STORY-031). When omitted, the chat
    # handler creates a new conversation and returns its id in
    # AgentChatResponse.conversation_id.
    conversation_id: str | None = None


# ── Agent tool result types ───────────────────────────────────────────────────


class NodeResult(BaseModel):
    id: str
    qualified_name: str | None = None
    label: str
    properties: dict = {}


class PathResult(BaseModel):
    nodes: list[NodeResult]
    hop_count: int


class ProvenancePayload(BaseModel):
    nodes: list[dict] = []
    edges: list[dict] = []
    queries_executed: list[str] = []
    nodes_used_in_response: list[str] = []
    total_nodes_traversed: int = 0
    reasoning_steps: int = 0
    tools_called: list[str] = []
    # One entry per dispatched tool call. Each is {"id", "name", "node_count"}.
    # Lets the frontend reconstruct the LLM's actual call chain instead of
    # just seeing the flat tool-name list. ``id`` mirrors the LLM-provided
    # tool_call_id when the native tool-use protocol is in use; falls back
    # to a synthetic id for legacy (direct-mode graph_search) dispatches.
    tool_calls: list[dict] = []


class AgentChatResponse(BaseModel):
    response: str
    session_id: str | None = None
    # Persisted conversation this turn was written to (STORY-031).
    conversation_id: str | None = None
    provenance: ProvenancePayload
