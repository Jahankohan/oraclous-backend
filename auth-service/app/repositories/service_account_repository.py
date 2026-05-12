"""Repository for AgentServiceAccount key lifecycle.

Handles create / validate / revoke for agent_service_account_keys table.
Key generation happens here — raw key is returned ONCE at creation, never stored.
"""

import secrets
import uuid
from datetime import datetime, timezone
from typing import Optional

from passlib.context import CryptContext
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker

from app.models.base_model import Base
from app.models.service_account_model import AgentServiceAccountKey

_bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# base62 alphabet for osk_ key generation
_BASE62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _generate_api_key() -> tuple[str, str]:
    """Generate osk_<base62(32 random bytes)>.

    Returns (raw_key, key_prefix) where key_prefix is the first 12 chars.
    """
    raw_bytes = secrets.token_bytes(32)
    n = int.from_bytes(raw_bytes, "big")
    chars = []
    while n:
        chars.append(_BASE62[n % 62])
        n //= 62
    token = "".join(reversed(chars)).rjust(43, "0")
    raw_key = f"osk_{token}"
    key_prefix = raw_key[:12]  # "osk_" + first 8 chars
    return raw_key, key_prefix


class ServiceAccountRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = sessionmaker(
            bind=self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def create_key(
        self,
        service_account_id: str,
        created_by_user_id: str,
        expires_at: Optional[datetime] = None,
    ) -> tuple[str, "AgentServiceAccountKey"]:
        """Generate a new API key, hash it, store it, return (raw_key, record).

        The raw key is ONLY available from this return value — never stored in DB.
        """
        raw_key, key_prefix = _generate_api_key()
        key_hash = _bcrypt_context.hash(raw_key)

        record = AgentServiceAccountKey(
            key_id=str(uuid.uuid4()),
            service_account_id=service_account_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            status="active",
            created_by_user_id=created_by_user_id,
            expires_at=expires_at,
        )
        async with self.Session() as session:
            session.add(record)
            await session.commit()
            await session.refresh(record)

        return raw_key, record

    async def validate_key(self, api_key: str) -> Optional[str]:
        """Validate an API key and return service_account_id if valid, else None.

        Lookup is optimized via key_prefix index; bcrypt verify runs only on candidates.
        """
        if not api_key.startswith("osk_") or len(api_key) < 12:
            return None

        key_prefix = api_key[:12]
        async with self.Session() as session:
            result = await session.execute(
                select(AgentServiceAccountKey).where(
                    AgentServiceAccountKey.key_prefix == key_prefix,
                    AgentServiceAccountKey.status == "active",
                )
            )
            candidates = result.scalars().all()

        for candidate in candidates:
            if candidate.expires_at and candidate.expires_at < datetime.now(
                timezone.utc
            ):
                continue
            if _bcrypt_context.verify(api_key, candidate.key_hash):
                return candidate.service_account_id

        return None

    async def revoke_keys_for_sa(self, service_account_id: str) -> int:
        """Revoke all active keys for a service account. Returns count revoked."""
        now = datetime.now(timezone.utc)
        async with self.Session() as session:
            result = await session.execute(
                update(AgentServiceAccountKey)
                .where(
                    AgentServiceAccountKey.service_account_id == service_account_id,
                    AgentServiceAccountKey.status == "active",
                )
                .values(status="revoked", revoked_at=now)
            )
            await session.commit()
            return result.rowcount

    async def has_active_key(self, service_account_id: str) -> bool:
        """Return True if the SA has at least one active, non-expired key."""
        async with self.Session() as session:
            result = await session.execute(
                select(AgentServiceAccountKey).where(
                    AgentServiceAccountKey.service_account_id == service_account_id,
                    AgentServiceAccountKey.status == "active",
                )
            )
            key = result.scalars().first()
        if key is None:
            return False
        if key.expires_at and key.expires_at < datetime.now(timezone.utc):
            return False
        return True

    async def update_last_used(self, service_account_id: str) -> None:
        """Fire-and-forget: update last_used_at for the SA's most recent active key."""
        try:
            now = datetime.now(timezone.utc)
            async with self.Session() as session:
                result = await session.execute(
                    select(AgentServiceAccountKey)
                    .where(
                        AgentServiceAccountKey.service_account_id == service_account_id,
                        AgentServiceAccountKey.status == "active",
                    )
                    .limit(1)
                )
                key = result.scalars().first()
                if key:
                    await session.execute(
                        update(AgentServiceAccountKey)
                        .where(AgentServiceAccountKey.key_id == key.key_id)
                        .values(last_used_at=now)
                    )
                    await session.commit()
        except Exception:
            pass  # Non-blocking — never let this fail a request
