from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from app.models.credential_model import UserCredential
from app.models.base_model import Base
from app.schema.credential_schema import CreateCredential, RequestCredentials, RequestCredentialsResponse, CredentialsUpdate
from app.models.enums import CredentialType
from typing import Optional, List
from uuid import UUID

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

    async def create_credential(self, user_credential: CreateCredential) -> UserCredential:
        cred = UserCredential(
            name=user_credential.name,
            provider=user_credential.provider,
            user_id=user_credential.user_id,
            tool_id=user_credential.tool_id,
            encrypted_cred=user_credential.credential,
            cred_type=CredentialType(user_credential.cred_type),
        )
        async with self.Session() as session:
            async with session.begin():
                session.add(cred)
            await session.refresh(cred)
            return cred

    async def list_credentials(self, cred_request: RequestCredentials) -> List[UserCredential]:
        async with self.Session() as session:
            stmt = select(UserCredential).where(UserCredential.user_id == cred_request.user_id)
            if cred_request.tool_id:
                stmt = stmt.where(UserCredential.tool_id == cred_request.tool_id)
            result = await session.execute(stmt)
            return result.scalars().all()

    async def get_credential_by_id(self, cred_id: UUID) -> Optional[UserCredential]:
        async with self.Session() as session:
            result = await session.execute(select(UserCredential).where(UserCredential.id == cred_id))
            return result.scalars().first()

    async def update_credential(self, u: CredentialsUpdate) -> UserCredential:
        async with self.Session.begin() as session:
            obj = await session.get(UserCredential, u.id)
            if not obj:
                return None

            if u.name is not None:       obj.name = u.name
            if u.provider is not None:   obj.provider = u.provider
            if u.user_id is not None:    obj.user_id = u.user_id
            if u.tool_id is not None:    obj.tool_id = u.tool_id
            if u.cred_type is not None:  obj.cred_type = CredentialType(u.cred_type)
            if u.credential is not None: obj.encrypted_cred = u.credential  # already encrypted

        # committed on exit; optionally re-load if your session expires on commit
        async with self.Session() as session:
            return await session.get(UserCredential, u.id)

    async def delete_credential(self, cred_id: UUID) -> bool:
        async with self.Session() as session:
            cred = await session.execute(select(UserCredential).where(UserCredential.id == cred_id))
            cred = cred.scalars().first()
            if cred:
                await session.delete(cred)
                await session.commit()
                return True
            return False
