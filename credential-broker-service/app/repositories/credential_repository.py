from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy.dialects.postgresql import insert
from typing import Optional, List
from uuid import UUID

from app.models.base_model import Base
from app.models.credential_model import Credential


class CredentialRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self):
        """Close the database engine."""
        await self.engine.dispose()

    async def create(self, provider: str, type_: str, encrypted_data: str, metadata: dict, created_by: UUID) -> Credential:
        """Insert a new credential or update if provider/type already exists."""
        async with self.Session() as session:
            stmt = (
                insert(Credential)
                .values(provider=provider, type=type_, encrypted_data=encrypted_data, cred_metadata=metadata, created_by=created_by)
                .returning(Credential)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.scalars().first()

    async def get(self, credential_id: UUID) -> Optional[Credential]:
        """Retrieve a credential by ID."""
        async with self.Session() as session:
            result = await session.execute(
                select(Credential).where(Credential.id == credential_id)
            )
            return result.scalars().first()

    async def list_all(self) -> List[Credential]:
        """List all stored credentials."""
        async with self.Session() as session:
            result = await session.execute(select(Credential))
            return result.scalars().all()

    async def list_by_user(self, user_id: UUID) -> List[Credential]:
        """List credentials for a specific user."""
        async with self.Session() as session:
            result = await session.execute(
                select(Credential).where(Credential.created_by == user_id)
            )
            return result.scalars().all()

    async def update(self, credential_id: UUID, provider: str, cred_type: str, encrypted_data: str, metadata: dict) -> Optional[Credential]:
        """Update credential (provider, type, and encrypted data)."""
        async with self.Session() as session:
            result = await session.execute(select(Credential).where(Credential.id == credential_id))
            credential = result.scalars().first()
            if not credential:
                return None
            credential.provider = provider
            credential.type = cred_type
            credential.encrypted_data = encrypted_data
            credential.cred_metadata = metadata
            await session.commit()
            await session.refresh(credential)
            return credential

    async def update_encrypted_data(self, credential_id: UUID, encrypted_data: str) -> Optional[Credential]:
        """Update only the encrypted data (e.g., after token refresh)."""
        async with self.Session() as session:
            result = await session.execute(select(Credential).where(Credential.id == credential_id))
            credential = result.scalars().first()
            if not credential:
                return None
            credential.encrypted_data = encrypted_data
            await session.commit()
            await session.refresh(credential)
            return credential

    async def delete(self, credential_id: UUID) -> bool:
        """Delete a credential by ID."""
        async with self.Session() as session:
            result = await session.execute(
                select(Credential).where(Credential.id == credential_id)
            )
            cred = result.scalars().first()
            if not cred:
                return False

            await session.delete(cred)
            await session.commit()
            return True
