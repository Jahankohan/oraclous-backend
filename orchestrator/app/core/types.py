from typing import List, Optional, Literal, Union, Dict, Any
from pydantic import BaseModel


class Suggestion(BaseModel):
    action: Literal["install", "connect"]
    mcp: Optional[str] = None
    provider: Optional[str] = None
    description: str


class MCPToolNode(BaseModel):
    id: str
    type: Literal["mcp_tool"]
    mcp: str
    tool: str
    params: Dict[str, Any]


class NativeToolNode(BaseModel):
    id: str
    type: Literal["native_tool"]
    tool: str
    params: Dict[str, Any]


class LLMNode(BaseModel):
    id: str
    type: Literal["llm"]
    prompt: str


class MissingToolNode(BaseModel):
    id: str
    type: Literal["missing_tool"]
    description: str
    suggestions: List[Suggestion]


ToolNode = Union[MCPToolNode, NativeToolNode, LLMNode, MissingToolNode]


class Edge(BaseModel):
    from_: str
    to: str


class WorkflowDraft(BaseModel):
    name: str
    description: Optional[str]
    inputs: Dict[str, str]
    nodes: List[ToolNode]
    edges: List[Edge]


class NodeInspection(BaseModel):
    node_id: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    timestamp: str


class Checkpoint(BaseModel):
    node_id: str
    state_snapshot: Dict[str, Any]
    created_at: str


class WorkflowExecutionState(BaseModel):
    workflow_id: str
    current_node: Optional[str]
    memory: Dict[str, Any]  # Shared context between nodes
    inspections: List[NodeInspection]  # For introspection before/after each node
    errors: List[Dict[str, Any]]  # Structured error logs
    artifacts: Dict[str, Any]  # Node outputs
    checkpoints: List[Checkpoint]  # Saved state for retry/resume
    started_at: str
    updated_at: Optional[str]
