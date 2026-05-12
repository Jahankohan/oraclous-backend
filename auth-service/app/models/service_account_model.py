"""SQLAlchemy model for agent service account API keys.

Stores bcrypt-hashed API keys for AgentServiceAccount principals.
The raw API key is NEVER stored — only the bcrypt hash and key_prefix (first 8 chars).
"""

import uuid

from sqlalchemy import Column, String, TIMESTAMP, func

from app.models.base_model import Base


class AgentServiceAccountKey(Base):
    __tablename__ = "agent_service_account_keys"

    key_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    service_account_id = Column(String, nullable=False, index=True)
    key_hash = Column(String, nullable=False)  # bcrypt(api_key, cost=12)
    key_prefix = Column(String, nullable=False, index=True)  # first 12 chars for lookup
    status = Column(String, nullable=False, default="active")  # active | revoked
    # JWT claim metadata — stored at creation for token exchange (no Neo4j in auth-service)
    tenant_id = Column(String, nullable=True)
    home_graph_id = Column(String, nullable=True)
    created_at = Column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at = Column(TIMESTAMP(timezone=True), nullable=True)
    revoked_at = Column(TIMESTAMP(timezone=True), nullable=True)
    last_used_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_by_user_id = Column(String, nullable=False)
