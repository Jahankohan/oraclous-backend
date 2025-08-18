from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.workflow import WorkflowDB, WorkflowExecutionDB, WorkflowTemplateDB, WorkflowShareDB
from app.schemas.workflow import CreateWorkflowRequest

class WorkflowRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_workflow(self, request: CreateWorkflowRequest) -> WorkflowDB:
        workflow = WorkflowDB(**request.dict())
        self.db.add(workflow)
        await self.db.commit()
        await self.db.refresh(workflow)
        return workflow

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowDB]:
        result = await self.db.execute(select(WorkflowDB).where(WorkflowDB.id == workflow_id))
        return result.scalar_one_or_none()

    async def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> Optional[WorkflowDB]:
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            return None
        for key, value in updates.items():
            setattr(workflow, key, value)
        await self.db.commit()
        await self.db.refresh(workflow)
        return workflow

    async def list_workflows(self, user_id: str, filters: Optional[Dict[str, Any]] = None) -> List[WorkflowDB]:
        query = select(WorkflowDB).where(WorkflowDB.user_id == user_id)
        if filters:
            for key, value in filters.items():
                query = query.where(getattr(WorkflowDB, key) == value)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def create_execution(self, workflow_id: str, params: Dict[str, Any]) -> WorkflowExecutionDB:
        execution = WorkflowExecutionDB(workflow_id=workflow_id, **params)
        self.db.add(execution)
        await self.db.commit()
        await self.db.refresh(execution)
        return execution

    # Additional methods for templates, shares, etc. can be added as needed.
