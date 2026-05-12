"""Pydantic schemas for Graph-Native Agent CRUD and execution."""

from typing import Literal

from pydantic import BaseModel, field_validator

# Valid tool names — enforced at creation time so no invalid tool can be persisted.
VALID_TOOLS = frozenset(
    {
        "graph_search",
        "community_members",
        "neighbors",
        "degree_centrality",
        "shortest_path",
        "taint_trace",
        "temporal_slice",
    }
)

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


class AgentChatResponse(BaseModel):
    response: str
    session_id: str | None = None
    provenance: ProvenancePayload
