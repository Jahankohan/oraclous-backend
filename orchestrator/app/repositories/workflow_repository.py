from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
from app.models.workflow_model import Workflow
from app.models.base import Base
from app.schemas.workflow_schema import WorkflowCreate, WorkflowUpdate
from typing import List
from sqlalchemy.orm import sessionmaker
from uuid import UUID

class WorkflowRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=True, pool_size=10, max_overflow=20)  

        # Correctly bind the engine to the sessionmaker
        self.Session = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
    
    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self):
        # Close the connection to the database engine
        await self.engine.dispose()

    async def create_workflow(self, data: WorkflowCreate) -> Workflow:
        async with self.Session() as session:
            async with session.begin():
                new_workflow = Workflow(**data.dict(by_alias=True))
                session.add(new_workflow)
            await session.refresh(new_workflow)
            return new_workflow

    async def get_workflow(self, workflow_id: UUID) -> Workflow:
        async with self.Session() as session:
            stmt = select(Workflow).where(Workflow.id == workflow_id)
            result = await session.execute(stmt)
            return result.scalars().first()

    async def update_workflow(self, workflow_id: UUID, data: WorkflowUpdate) -> Workflow:
        async with self.Session() as session:
            stmt = select(Workflow).where(Workflow.id == workflow_id)
            result = await session.execute(stmt)
            workflow = result.scalars().first()
            if not workflow:
                return None
            for key, value in data.dict(exclude_unset=True, by_alias=True).items():
                setattr(workflow, key, value)
            await session.commit()
            await session.refresh(workflow)
            return workflow

    async def delete_workflow(self, workflow_id: UUID) -> bool:
        async with self.Session() as session:
            stmt = select(Workflow).where(Workflow.id == workflow_id)
            result = await session.execute(stmt)
            workflow = result.scalars().first()
            if not workflow:
                return False
            await session.delete(workflow)
            await session.commit()
            return True

    async def list_all_workflows(self) -> List[Workflow]:
        async with self.Session() as session:
            stmt = select(Workflow).order_by(Workflow.created_at.desc())
            result = await session.execute(stmt)
            return result.scalars().all()