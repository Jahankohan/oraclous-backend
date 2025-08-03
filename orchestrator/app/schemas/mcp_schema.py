from datetime import datetime
from pydantic import BaseModel
from uuid import UUID

class MCPCreate(BaseModel):
    url: str
    name: str
    description: str
    token: str
    category: str

class MCPCreated(MCPCreate):
    id: UUID
    created_at: datetime
    updated_at: datetime
