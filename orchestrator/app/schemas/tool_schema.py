from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class ToolCreate(BaseModel):
    name: str = Field(..., description="Unique name of the tool")
    description: Optional[str] = Field("", description="Short description of the tool")
    url: str = Field(..., description="MCP endpoint serving the tool")
    input_schema: Optional[dict] = Field(None, description="JSON Schema for tool input")
    output_example: Optional[dict] = Field(None, description="Example output for documentation")
    category: Optional[str] = Field(None, description="Inferred or user-defined category")
    tags: Optional[List[str]] = Field(default_factory=list)


class ToolUpdate(BaseModel):
    description: Optional[str]
    url: Optional[str]
    input_schema: Optional[dict]
    output_example: Optional[dict]
    category: Optional[str]
    tags: Optional[List[str]]


class ToolRead(ToolCreate):
    id: UUID
    created_at: datetime
    updated_at: datetime
