import uuid

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.database import Base


class Organization(Base):
    """First-class organization entity (TASK-201).

    Backed by both PostgreSQL (this table, source of truth for metadata)
    and a synced Neo4j ``:Organization`` node used for ReBAC ownership
    edges. ``org_id`` everywhere == ``str(Organization.id)``.
    """

    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    # URL-safe subdomain label — ``<slug>.oraclous.com``. Unique (the DB
    # constraint is the authoritative guarantee); generated from ``name`` at
    # creation time, see ``app/utils/slug.py``.
    slug = Column(String(63), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    # Organization logo (a URL). Rendered on org-scoped login / invitation
    # screens. Null until a logo is set.
    logo_url = Column(String(512), nullable=True)
    owner_user_id = Column(UUID(as_uuid=True), nullable=False)
    settings = Column(JSON, nullable=False, default=dict)
    status = Column(String(50), default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class OrgInvitation(Base):
    """A pending invitation for someone to join an organization as a member.

    An org owner or admin creates an invitation for an email address; the
    invitee receives a link carrying ``token`` and, once authenticated,
    accepts it — which registers them as an org member with ``org_role``.

    Postgres-only: the ``BELONGS_TO`` membership edge in Neo4j is created at
    accept time via ``org_member_service.add_member``. ``status`` is one of
    ``pending | accepted | revoked | expired``.
    """

    __tablename__ = "org_invitations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    email = Column(String(320), nullable=False)
    org_role = Column(String(64), nullable=False, default="member")
    token = Column(String(64), nullable=False, unique=True, index=True)
    status = Column(String(32), nullable=False, default="pending")
    # Optional per-subgraph access to apply when the invite is accepted —
    # {"role": <rebac role>, "graph_ids": [...] | "all"}. Used for `member`
    # invites (owner/admin get all subgraphs automatically). Null = none.
    subgraph_grants = Column(JSON, nullable=True)
    invited_by_user_id = Column(UUID(as_uuid=True), nullable=False)
    accepted_by_user_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    accepted_at = Column(DateTime(timezone=True), nullable=True)
