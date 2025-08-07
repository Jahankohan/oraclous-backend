from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, selectinload
from sqlalchemy.future import select
from app.models.ingestion_job_model import IngestionJob, IngestionStatus
from app.models.base import Base
from app.schemas.ingestion_job_schema import IngestionJobCreate, IngestionJobUpdate
from typing import List, Optional
from uuid import UUID

class IngestionJobRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)
    
    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        await self.engine.dispose()

    async def create(self, data: IngestionJobCreate) -> IngestionJob:
        async with self.Session() as session:
            async with session.begin():
                job = IngestionJob(**data.model_dump())
                session.add(job)
                await session.flush()
                await session.refresh(job)
                return job

    async def get_by_id(self, job_id: UUID) -> Optional[IngestionJob]:
        async with self.Session() as session:
            result = await session.execute(
                select(IngestionJob)
                .options(selectinload(IngestionJob.source))
                .where(IngestionJob.id == job_id)
            )
            return result.scalars().first()

    async def get_by_source(self, source_id: UUID) -> List[IngestionJob]:
        async with self.Session() as session:
            result = await session.execute(
                select(IngestionJob)
                .where(IngestionJob.source_id == source_id)
                .order_by(IngestionJob.created_at.desc())
            )
            return result.scalars().all()

    async def get_by_status(self, status: IngestionStatus) -> List[IngestionJob]:
        async with self.Session() as session:
            result = await session.execute(
                select(IngestionJob)
                .where(IngestionJob.status == status)
                .order_by(IngestionJob.created_at.desc())
            )
            return result.scalars().all()

    async def update(self, job_id: UUID, data: IngestionJobUpdate) -> Optional[IngestionJob]:
        async with self.Session() as session:
            async with session.begin():
                result = await session.execute(select(IngestionJob).where(IngestionJob.id == job_id))
                job = result.scalars().first()
                if not job:
                    return None
                
                for key, value in data.model_dump(exclude_unset=True).items():
                    setattr(job, key, value)
                
                await session.flush()
                await session.refresh(job)
                return job

    async def delete(self, job_id: UUID) -> bool:
        async with self.Session() as session:
            async with session.begin():
                result = await session.execute(select(IngestionJob).where(IngestionJob.id == job_id))
                job = result.scalars().first()
                if not job:
                    return False
                
                await session.delete(job)
                return True

    async def list_all(self) -> List[IngestionJob]:
        async with self.Session() as session:
            result = await session.execute(
                select(IngestionJob)
                .options(selectinload(IngestionJob.source))
                .order_by(IngestionJob.created_at.desc())
            )
            return result.scalars().all()
