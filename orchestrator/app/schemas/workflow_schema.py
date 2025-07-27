from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from uuid import UUID

class NodeConfig(BaseModel):
    id: str
    type: str
    config: Optional[dict] = {}

class EdgeConfig(BaseModel):
    from_: str = Field(..., alias="from")
    to: str

class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    nodes: List[NodeConfig]
    edges: List[EdgeConfig]

class WorkflowUpdate(BaseModel):
    name: Optional[str]
    description: Optional[str]
    nodes: Optional[List[NodeConfig]]
    edges: Optional[List[EdgeConfig]]

class WorkflowRead(WorkflowCreate):
    id: UUID
