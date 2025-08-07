from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, selectinload
from sqlalchemy.future import select
from app.models.document_model import Document
from app.models.base import Base
from app.schemas.document_schema import DocumentCreate, DocumentUpdate
from typing import List, Optional
from uuid import UUID

class DocumentRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)
    
    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        await self.engine.dispose()

    async def create_many(self, documents: List[DocumentCreate]) -> List[Document]:
        async with self.Session() as session:
            async with session.begin():
                doc_objects = [Document(**doc.model_dump()) for doc in documents]
                session.add_all(doc_objects)
                await session.flush()
                for doc in doc_objects:
                    await session.refresh(doc)
                return doc_objects

    async def get_by_id(self, doc_id: UUID) -> Optional[Document]:
        async with self.Session() as session:
            result = await session.execute(
                select(Document)
                .options(selectinload(Document.source), selectinload(Document.job))
                .where(Document.id == doc_id)
            )
            return result.scalars().first()

    async def get_by_job(self, job_id: UUID) -> List[Document]:
        async with self.Session() as session:
            result = await session.execute(
                select(Document)
                .where(Document.job_id == job_id)
                .order_by(Document.created_at.desc())
            )
            return result.scalars().all()

    async def get_by_source(self, source_id: UUID) -> List[Document]:
        async with self.Session() as session:
            result = await session.execute(
                select(Document)
                .where(Document.source_id == source_id)
                .order_by(Document.created_at.desc())
            )
            return result.scalars().all()

    async def search_by_content(self, query: str, limit: int = 50) -> List[Document]:
        async with self.Session() as session:
            # Simple text search - in production you'd want full-text search
            result = await session.execute(
                select(Document)
                .where(Document.content.ilike(f"%{query}%"))
                .limit(limit)
                .order_by(Document.created_at.desc())
            )
            return result.scalars().all()

    async def update(self, doc_id: UUID, data: DocumentUpdate) -> Optional[Document]:
        async with self.Session() as session:
            async with session.begin():
                result = await session.execute(select(Document).where(Document.id == doc_id))
                document = result.scalars().first()
                if not document:
                    return None
                
                for key, value in data.model_dump(exclude_unset=True).items():
                    setattr(document, key, value)
                
                await session.flush()
                await session.refresh(document)
                return document

    async def delete(self, doc_id: UUID) -> bool:
        async with self.Session() as session:
            async with session.begin():
                result = await session.execute(select(Document).where(Document.id == doc_id))
                document = result.scalars().first()
                if not document:
                    return False
                
                await session.delete(document)
                return True

    async def delete_by_job(self, job_id: UUID) -> int:
        """Delete all documents for a job and return count deleted"""
        async with self.Session() as session:
            async with session.begin():
                result = await session.execute(select(Document).where(Document.job_id == job_id))
                documents = result.scalars().all()
                count = len(documents)
                
                for doc in documents:
                    await session.delete(doc)
                
                return count

    async def get_by_hash(self, content_hash: str) -> Optional[Document]:
        """Check for duplicate content"""
        async with self.Session() as session:
            result = await session.execute(
                select(Document).where(Document.content_hash == content_hash)
            )
            return result.scalars().first()

    async def list_all(self) -> List[Document]:
        async with self.Session() as session:
            result = await session.execute(
                select(Document)
                .options(selectinload(Document.source), selectinload(Document.job))
                .order_by(Document.created_at.desc())
            )
            return result.scalars().all()
