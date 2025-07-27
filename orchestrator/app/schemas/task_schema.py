from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.task_model import TaskStatus


class TaskCreate(BaseModel):
    workflow_id: UUID
    inputs: Optional[dict] = {}
    owner_id: Optional[str] = None


class TaskUpdate(BaseModel):
    status: Optional[TaskStatus]
    outputs: Optional[dict]
    logs: Optional[dict]
    token_usage: Optional[str]
    metadata: Optional[dict]
    finished_at: Optional[datetime]


class TaskRead(BaseModel):
    id: UUID
    workflow_id: UUID
    status: TaskStatus
    inputs: Optional[dict]
    outputs: Optional[dict]
    logs: Optional[dict]
    token_usage: Optional[str]
    metadata: Optional[dict]
    started_at: datetime
    finished_at: Optional[datetime]
    owner_id: Optional[str]

    class Config:
        orm_mode = True
