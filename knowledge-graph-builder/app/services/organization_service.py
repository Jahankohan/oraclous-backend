"""Organization service (TASK-201).

Coordinates the two-store representation of an organization:

  - PostgreSQL ``organizations`` table — source of truth for metadata.
  - Neo4j ``:Organization:__Platform__`` node — carries ReBAC ownership
    edges (``:User-[:BELONGS_TO {org_role}]->:Organization``).

``org_id`` everywhere == ``str(Organization.id)``.

All Cypher is parameterized — user input is never string-interpolated.
"""

from __future__ import annotations

import uuid

from neo4j import AsyncDriver
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.organization import Organization

logger = get_logger(__name__)

# Explicit allowlist for update_organization — the single source of truth for
# which SQL columns may be patched. Any key outside this set is rejected.
_ALLOWED_ORG_UPDATE_FIELDS: frozenset[str] = frozenset(
    {"name", "description", "settings"}
)


async def create_organization(
    db: AsyncSession,
    driver: AsyncDriver,
    *,
    name: str,
    description: str,
    settings: dict,
    owner_user_id: str,
) -> Organization:
    """Create an organization in PostgreSQL, then mirror it into Neo4j.

    Best-effort two-store consistency: if the Neo4j write fails, the SQL row
    inserted in step 1 is deleted before the error is re-raised.
    """
    # 1. Insert the SQL row (source of truth) and commit.
    organization = Organization(
        name=name,
        description=description,
        owner_user_id=owner_user_id,
        settings=settings or {},
        status="active",
    )
    db.add(organization)
    await db.commit()
    await db.refresh(organization)

    org_id = str(organization.id)

    # 2. Mirror into Neo4j: :Organization node + owner :User + BELONGS_TO edge.
    try:
        async with driver.session() as session:
            await session.run(
                """
                MERGE (o:Organization:__Platform__ {org_id: $org_id})
                ON CREATE SET
                    o.graph_id    = "__system__",
                    o.name        = $name,
                    o.description = $description,
                    o.status      = "active",
                    o.created_at  = datetime()
                MERGE (u:User:__Platform__ {
                    user_id: $owner_user_id, graph_id: "__system__"
                })
                MERGE (u)-[r:BELONGS_TO]->(o)
                ON CREATE SET
                    r.org_role = "owner",
                    r.since    = datetime()
                """,
                {
                    "org_id": org_id,
                    "name": name,
                    "description": description,
                    "owner_user_id": owner_user_id,
                },
            )
    except Exception:
        # Roll back the SQL insert to avoid an orphan row with no Neo4j node.
        logger.exception(
            "Neo4j sync failed for organization %s — rolling back SQL row", org_id
        )
        await db.delete(organization)
        await db.commit()
        raise

    return organization


async def get_organization(db: AsyncSession, org_id: str) -> Organization | None:
    """Fetch an organization by id.

    Returns None if it does not exist, or if ``org_id`` is not a valid UUID —
    so a malformed path parameter yields a clean 404, never a 500.
    """
    try:
        org_uuid = uuid.UUID(org_id)
    except (ValueError, TypeError):
        return None
    result = await db.execute(select(Organization).where(Organization.id == org_uuid))
    return result.scalar_one_or_none()


async def list_organizations(
    db: AsyncSession, owner_user_id: str
) -> list[Organization]:
    """List all organizations owned by the given user."""
    result = await db.execute(
        select(Organization)
        .where(Organization.owner_user_id == owner_user_id)
        .order_by(Organization.created_at)
    )
    return list(result.scalars().all())


async def get_or_create_default_org(
    db: AsyncSession,
    driver: AsyncDriver,
    user_id: str,
) -> str:
    """Return the ``org_id`` of the user's default (personal) organization.

    The default organization is defined as the *oldest* ``Organization`` row
    the user owns. If the user owns none, one is created via
    :func:`create_organization` (name ``"Personal Organization"``, empty
    description and settings, owner = ``user_id``).

    Idempotent: once a personal organization exists, repeated calls return its
    id without creating duplicates.
    """
    result = await db.execute(
        select(Organization)
        .where(Organization.owner_user_id == user_id)
        .order_by(Organization.created_at)
        .limit(1)
    )
    existing = result.scalar_one_or_none()
    if existing is not None:
        return str(existing.id)

    organization = await create_organization(
        db,
        driver,
        name="Personal Organization",
        description="",
        settings={},
        owner_user_id=user_id,
    )
    return str(organization.id)


async def list_org_graphs(
    driver: AsyncDriver,
    org_id: str,
) -> list[dict]:
    """Return the :Graph:__Platform__ nodes owned by *org_id*.

    Soft-deleted graphs (``status == 'deactivated'``) are excluded. The caller
    is responsible for verifying that the requester owns the organization.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (g:Graph:__Platform__ {org_id: $org_id})
            WHERE coalesce(g.status, 'active') <> 'deactivated'
            RETURN g {
                .graph_id,
                .name,
                .description,
                .user_id,
                .org_id,
                .created_at,
                .updated_at,
                .node_count,
                .relationship_count,
                .status,
                .federatable,
                .federation_group
            } AS graph
            ORDER BY g.created_at DESC
            """,
            {"org_id": org_id},
        )
        return [dict(record["graph"]) async for record in result]


async def update_organization(
    db: AsyncSession,
    driver: AsyncDriver,
    org_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    settings: dict | None = None,
) -> Organization | None:
    """Patch an organization's SQL row and mirror name/description onto Neo4j.

    Returns the updated organization, or None if it does not exist.
    """
    organization = await get_organization(db, org_id)
    if organization is None:
        return None

    updates: dict = {}
    if name is not None:
        updates["name"] = name
    if description is not None:
        updates["description"] = description
    if settings is not None:
        updates["settings"] = settings

    # Structural guard against accidental column writes.
    unknown = set(updates) - _ALLOWED_ORG_UPDATE_FIELDS
    if unknown:
        raise ValueError(f"Disallowed organization update fields: {unknown!r}")

    for field, value in updates.items():
        setattr(organization, field, value)

    await db.commit()
    await db.refresh(organization)

    # Mirror name/description/status onto the Neo4j node.
    async with driver.session() as session:
        await session.run(
            """
            MATCH (o:Organization {org_id: $org_id})
            SET o.name        = $name,
                o.description = $description,
                o.status      = $status
            """,
            {
                "org_id": org_id,
                "name": organization.name,
                "description": organization.description,
                "status": organization.status,
            },
        )

    return organization
