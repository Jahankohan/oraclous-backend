from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from app.models.data_source_model import DataSource
from app.models.base import Base
from app.schemas.data_source_schema import DataSourceCreate, DataSourceUpdate
from typing import List, Optional
from uuid import UUID

class DataSourceRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)
    
    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        await self.engine.dispose()

    async def create(self, data: DataSourceCreate) -> DataSource:
        async with self.Session() as session:
            async with session.begin():
                source = DataSource(**data.model_dump())
                session.add(source)
                await session.flush()
                await session.refresh(source)
                return source

    async def get_by_id(self, source_id: UUID) -> Optional[DataSource]:
        async with self.Session() as session:
            result = await session.execute(select(DataSource).where(DataSource.id == source_id))
            return result.scalars().first()

    async def get_by_owner(self, owner_id: str) -> List[DataSource]:
        async with self.Session() as session:
            result = await session.execute(
                select(DataSource).where(DataSource.owner_id == owner_id)
            )
            return result.scalars().all()

    async def update(self, source_id: UUID, data: DataSourceUpdate) -> Optional[DataSource]:
        async with self.Session() as session:
            async with session.begin():
                result = await session.execute(select(DataSource).where(DataSource.id == source_id))
                source = result.scalars().first()
                if not source:
                    return None
                
                for key, value in data.model_dump(exclude_unset=True).items():
                    setattr(source, key, value)
                
                await session.flush()
                await session.refresh(source)
                return source

    async def delete(self, source_id: UUID) -> bool:
        async with self.Session() as session:
            async with session.begin():
                result = await session.execute(select(DataSource).where(DataSource.id == source_id))
                source = result.scalars().first()
                if not source:
                    return False
                
                await session.delete(source)
                return True

    async def list_all(self) -> List[DataSource]:
        async with self.Session() as session:
            result = await session.execute(select(DataSource).order_by(DataSource.created_at.desc()))
            return result.scalars().all()
