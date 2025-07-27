from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from app.models.task_model import Task
from app.schemas.task_schema import TaskCreate, TaskUpdate
from typing import List, Optional
from app.models.base import Base
from uuid import UUID

class TaskRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)
    
    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self):
        # Close the connection to the database engine
        await self.engine.dispose()

    async def create_task(self, data: TaskCreate) -> Task:
        async with self.session_factory() as session:
            async with session.begin():
                task = Task(**data.model_dump())
                session.add(task)
                await session.flush()
                await session.refresh(task)
                return task

    async def get_task(self, task_id: UUID) -> Optional[Task]:
        async with self.session_factory() as session:
            stmt = select(Task).where(Task.id == task_id)
            result = await session.execute(stmt)
            return result.scalars().first()

    async def update_task(self, task_id: UUID, data: TaskUpdate) -> Optional[Task]:
        async with self.session_factory() as session:
            async with session.begin():
                stmt = select(Task).where(Task.id == task_id)
                result = await session.execute(stmt)
                task = result.scalars().first()
                if not task:
                    return None
                for key, value in data.model_dump(exclude_unset=True).items():
                    setattr(task, key, value)
                await session.flush()
                await session.refresh(task)
                return task

    async def delete_task(self, task_id: UUID) -> bool:
        async with self.session_factory() as session:
            async with session.begin():
                stmt = select(Task).where(Task.id == task_id)
                result = await session.execute(stmt)
                task = result.scalars().first()
                if not task:
                    return False
                await session.delete(task)
                return True

    async def list_all_tasks(self) -> List[Task]:
        async with self.session_factory() as session:
            stmt = select(Task).order_by(Task.created_at.desc())
            result = await session.execute(stmt)
            return result.scalars().all()
