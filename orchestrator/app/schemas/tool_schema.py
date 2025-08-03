from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class ToolCreate(BaseModel):
    mcp_server_id: UUID
    name: str
    description: Optional[str]
    input_schema: Optional[dict]
    output_example: Optional[dict]


class ToolUpdate(BaseModel):
    description: Optional[str]
    input_schema: Optional[dict]
    output_example: Optional[dict]

class ToolRead(ToolCreate):
    id: UUID
    created_at: datetime
    updated_at: datetime
