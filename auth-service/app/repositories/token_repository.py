from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, List
from datetime import datetime, timedelta
import logging

from app.models.base_model import Base
from app.models.token_model import OauthTokens

logger = logging.getLogger(__name__)

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
        """Get OAuth token for a user and provider."""
        try:
            async with self.Session() as session:
                result = await session.execute(
                    select(OauthTokens).where(
                        OauthTokens.user_id == user_id, 
                        OauthTokens.provider == provider
                    )
                )
                token = result.scalars().first()
                
                if token:
                    logger.debug(f"Retrieved token for user {user_id}, provider {provider}")
                else:
                    logger.info(f"No token found for user {user_id}, provider {provider}")
                    
                return token
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving token for {user_id}/{provider}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving token for {user_id}/{provider}: {e}")
            raise
    
    async def save_token(
        self, 
        user_id: str, 
        provider: str, 
        access_token: str, 
        refresh_token: Optional[str], 
        scopes: List[str], 
        expires_at: Optional[datetime]
    ):
        """Insert or update token while merging scopes."""
        try:
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
                
                logger.info(f"Saved token for user {user_id}, provider {provider} with scopes: {list(merged_scopes)}")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error saving token for {user_id}/{provider}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving token for {user_id}/{provider}: {e}")
            raise

    async def update_access_token(self, user_id: str, provider: str, access_token: str, expires_at: datetime):
        """Update only the access token and expiration."""
        try:
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
                    logger.info(f"Updated access token for user {user_id}, provider {provider}")
                else:
                    logger.warning(f"No token found to update for user {user_id}, provider {provider}")
                    
        except SQLAlchemyError as e:
            logger.error(f"Database error updating access token for {user_id}/{provider}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating access token for {user_id}/{provider}: {e}")
            raise
    
    async def list_tokens(self, user_id: str) -> List[OauthTokens]:
        """List all tokens for a user across providers."""
        try:
            async with self.Session() as session:
                result = await session.execute(
                    select(OauthTokens).where(OauthTokens.user_id == user_id)
                )
                tokens = result.scalars().all()
                logger.debug(f"Retrieved {len(tokens)} tokens for user {user_id}")
                return tokens
                
        except SQLAlchemyError as e:
            logger.error(f"Database error listing tokens for user {user_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing tokens for user {user_id}: {e}")
            raise

    async def delete_token(self, user_id: str, provider: str) -> bool:
        """Delete a token for a user and provider."""
        try:
            async with self.Session() as session:
                result = await session.execute(
                    select(OauthTokens).where(
                        OauthTokens.user_id == user_id,
                        OauthTokens.provider == provider
                    )
                )
                token = result.scalars().first()
                
                if token:
                    await session.delete(token)
                    await session.commit()
                    logger.info(f"Deleted token for user {user_id}, provider {provider}")
                    return True
                else:
                    logger.warning(f"No token found to delete for user {user_id}, provider {provider}")
                    return False
                    
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting token for {user_id}/{provider}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting token for {user_id}/{provider}: {e}")
            raise

    async def get_tokens_expiring_soon(self, minutes: int = 30) -> List[OauthTokens]:
        """Get tokens that will expire within the specified minutes."""
        try:
            cutoff_time = datetime.utcnow() + timedelta(minutes=minutes)
            
            async with self.Session() as session:
                result = await session.execute(
                    select(OauthTokens).where(
                        OauthTokens.expires_at.isnot(None),
                        OauthTokens.expires_at <= cutoff_time,
                        OauthTokens.refresh_token.isnot(None)
                    )
                )
                tokens = result.scalars().all()
                logger.debug(f"Found {len(tokens)} tokens expiring within {minutes} minutes")
                return tokens
                
        except SQLAlchemyError as e:
            logger.error(f"Database error getting expiring tokens: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting expiring tokens: {e}")
            raise

    async def get_user_providers(self, user_id: str) -> List[str]:
        """Get list of providers that user has tokens for."""
        try:
            async with self.Session() as session:
                result = await session.execute(
                    select(OauthTokens.provider).where(OauthTokens.user_id == user_id)
                )
                providers = [row[0] for row in result.fetchall()]
                logger.debug(f"User {user_id} has tokens for providers: {providers}")
                return providers
                
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user providers for {user_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting user providers for {user_id}: {e}")
            raise

    async def validate_token_scopes(self, user_id: str, provider: str, required_scopes: List[str]) -> dict:
        """Check if user's token has all required scopes."""
        try:
            token = await self.get_token(user_id, provider)
            
            if not token:
                return {
                    "valid": False,
                    "missing_scopes": required_scopes,
                    "current_scopes": [],
                    "token_exists": False
                }
            
            current_scopes = set(token.scopes or [])
            missing_scopes = [scope for scope in required_scopes if scope not in current_scopes]
            
            is_expired = token.expires_at and datetime.utcnow() > token.expires_at
            
            return {
                "valid": len(missing_scopes) == 0 and not is_expired,
                "missing_scopes": missing_scopes,
                "current_scopes": list(current_scopes),
                "token_exists": True,
                "token_expired": is_expired
            }
            
        except Exception as e:
            logger.error(f"Error validating token scopes for {user_id}/{provider}: {e}")
            raise