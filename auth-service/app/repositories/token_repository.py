from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.future import select
from typing import Optional, List
from datetime import datetime

from app.models.base_model import Base
from app.models.token_model import OauthTokens

class TokenRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)
    
    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        # Close the connection to the database engine
        await self.engine.dispose()
    
    async def get_token(self, user_id: str, provider: str) -> Optional[OauthTokens]:
        async with self.Session() as session:
            result = await session.execute(select(OauthTokens).where(OauthTokens.user_id == user_id, OauthTokens.provider == provider))
            return result.scalars().first()
    
    async def save_token(self, user_id: str, provider: str, access_token: str, refresh_token: Optional[str], scopes: List[str], expires_at: Optional[datetime]):
        """Insert or update token while merging scopes."""
        async with self.Session() as session:
            # Fetch existing token to merge scopes
            result = await session.execute(
                select(OauthTokens).where(
                    OauthTokens.user_id == user_id,
                    OauthTokens.provider == provider
                )
            )
            existing_token = result.scalars().first()

            merged_scopes = set(scopes)
            if existing_token and existing_token.scopes:
                merged_scopes |= set(existing_token.scopes)

            stmt = insert(OauthTokens).values(
                user_id=user_id,
                provider=provider,
                access_token=access_token,
                refresh_token=refresh_token,
                scopes=list(merged_scopes),
                expires_at=expires_at
            ).on_conflict_do_update(
                index_elements=["user_id", "provider"],
                set_={
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "scopes": list(merged_scopes),
                    "expires_at": expires_at
                }
            )
            await session.execute(stmt)
            await session.commit()


    async def update_access_token(self, user_id: str, provider: str, access_token: str, expires_at: datetime):
        """Update only the access token and expiration."""
        async with self.Session() as session:
            result = await session.execute(
                select(OauthTokens).where(
                    OauthTokens.user_id == user_id,
                    OauthTokens.provider == provider
                )
            )
            token = result.scalars().first()
            if token:
                token.access_token = access_token
                token.expires_at = expires_at
                session.add(token)
                await session.commit()
    
    async def list_tokens(self, user_id: str) -> List[OauthTokens]:
        """List all tokens for a user across providers."""
        async with self.Session() as session:
            result = await session.execute(
                select(OauthTokens).where(OauthTokens.user_id == user_id)
            )
            return result.scalars().all()
