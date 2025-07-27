from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import NoResultFound
from uuid import UUID
from app.models.workflow_model import Workflow
from app.schemas.workflow_schema import WorkflowCreate, WorkflowUpdate
from typing import List
from app.repositories.workflow_repository import WorkflowRepository

class WorkflowService:
    def __init__(self, repository: WorkflowRepository):
        self.repository = repository
    
    async def create_workflow(self, data: WorkflowCreate) -> Workflow:
        return await self.repository.create_workflow(data)

    async def get_workflow_by_id(self, workflow_id: UUID) -> Workflow:
        return await self.repository.get_workflow(workflow_id)

    async def update_workflow_by_id(self, workflow_id: UUID, data: WorkflowUpdate) -> Workflow:
        return await self.repository.update_workflow(workflow_id, data)

    async def delete_workflow_by_id(self, workflow_id: UUID) -> None:
        await self.repository.delete_workflow(workflow_id)

    async def list_all_workflows(self) -> List[Workflow]:
        return await self.repository.list_all_workflows()
