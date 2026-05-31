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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.organization import Organization
from app.utils.slug import generate_unique_slug, is_reserved

logger = get_logger(__name__)

# Explicit allowlist for update_organization — the single source of truth for
# which SQL columns may be patched. Any key outside this set is rejected.
_ALLOWED_ORG_UPDATE_FIELDS: frozenset[str] = frozenset(
    {"name", "description", "settings", "logo_url"}
)


async def create_organization(
    db: AsyncSession,
    driver: AsyncDriver,
    *,
    name: str,
    description: str,
    settings: dict,
    owner_user_id: str,
    slug: str | None = None,
    logo_url: str | None = None,
) -> Organization:
    """Create an organization in PostgreSQL, then mirror it into Neo4j.

    The subdomain ``slug`` is auto-generated from ``name`` when not supplied.
    An explicitly supplied slug is rejected (``ValueError``) when reserved or
    already taken. The DB unique constraint is the authoritative guarantee — a
    concurrent-create collision (``IntegrityError``) is retried with a fresh
    auto-generated slug.

    Best-effort two-store consistency: if the Neo4j write fails, the SQL row
    inserted in step 2 is deleted before the error is re-raised.
    """
    # 1. Resolve the slug. The DB unique index is authoritative; this set is
    #    just a best-effort first pick.
    taken: set[str] = set((await db.execute(select(Organization.slug))).scalars().all())
    if slug:
        slug = slug.lower()
        if is_reserved(slug):
            raise ValueError(f"Slug '{slug}' is reserved")
        if slug in taken:
            raise ValueError(f"Slug '{slug}' is already taken")
        resolved_slug = slug
    else:
        resolved_slug = generate_unique_slug(name, taken.__contains__)

    # 2. Insert the SQL row (source of truth), retrying on a concurrent slug
    #    collision (another create committing the same slug between our SELECT
    #    and INSERT).
    organization: Organization | None = None
    for _attempt in range(5):
        organization = Organization(
            name=name,
            slug=resolved_slug,
            description=description,
            logo_url=logo_url,
            owner_user_id=owner_user_id,
            settings=settings or {},
            status="active",
        )
        db.add(organization)
        try:
            await db.commit()
            break
        except IntegrityError:
            await db.rollback()
            if slug is not None:
                # An explicitly requested slug lost the race — clean error.
                raise ValueError(f"Slug '{slug}' is already taken") from None
            taken.add(resolved_slug)
            resolved_slug = generate_unique_slug(name, taken.__contains__)
    else:
        raise ValueError("Could not allocate a unique organization slug")
    await db.refresh(organization)

    org_id = str(organization.id)

    # 3. Mirror into Neo4j: :Organization node + owner :User + BELONGS_TO edge.
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


async def get_organization_by_slug(db: AsyncSession, slug: str) -> Organization | None:
    """Fetch an organization by its subdomain slug (case-insensitive).

    Returns None if no organization has that slug.
    """
    result = await db.execute(
        select(Organization).where(Organization.slug == slug.lower())
    )
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


async def list_user_organizations(
    db: AsyncSession,
    driver: AsyncDriver,
    user_id: str,
) -> list[tuple[Organization, str]]:
    """List every organization the user belongs to, with their role.

    The set of memberships is read from Neo4j (the ``:User-[:BELONGS_TO]->
    :Organization`` edges), then the matching ``Organization`` SQL rows are
    loaded. Any ``org_id`` with a Neo4j edge but no SQL row is skipped
    defensively, so a partial two-store state never yields a 500.

    Returns ``[(Organization, org_role), ...]`` ordered by org creation time.
    """
    # 1. Read the user's memberships (org_id + role) from Neo4j.
    roles_by_org_id: dict[str, str] = {}
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (:User:__Platform__ {user_id: $uid})
                  -[r:BELONGS_TO]->(o:Organization:__Platform__)
            RETURN o.org_id AS org_id, r.org_role AS org_role
            """,
            {"uid": user_id},
        )
        async for record in result:
            org_id = record["org_id"]
            if org_id is not None:
                roles_by_org_id[org_id] = record["org_role"]

    if not roles_by_org_id:
        return []

    # 2. Parse org_ids to UUIDs (skip malformed ones defensively).
    org_uuids: list[uuid.UUID] = []
    for org_id in roles_by_org_id:
        try:
            org_uuids.append(uuid.UUID(org_id))
        except (ValueError, TypeError):
            logger.warning("Skipping malformed org_id from BELONGS_TO edge: %r", org_id)

    if not org_uuids:
        return []

    # 3. SQL-load the matching Organization rows.
    result = await db.execute(
        select(Organization)
        .where(Organization.id.in_(org_uuids))
        .order_by(Organization.created_at)
    )
    organizations = list(result.scalars().all())

    # 4. Pair each row with its role; skip any org_id with no SQL row.
    return [(org, roles_by_org_id[str(org.id)]) for org in organizations]


async def get_user_org_role(
    driver: AsyncDriver,
    org_id: str,
    user_id: str,
) -> str | None:
    """Return the caller's ``org_role`` on the org, or None if not a member.

    Reads the ``:User-[:BELONGS_TO]->:Organization`` edge directly from Neo4j.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (:User:__Platform__ {user_id: $uid})
                  -[r:BELONGS_TO]->(:Organization:__Platform__ {org_id: $org_id})
            RETURN r.org_role AS org_role
            """,
            {"uid": user_id, "org_id": org_id},
        )
        record = await result.single()
    return record["org_role"] if record is not None else None


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


async def list_member_org_graphs(
    driver: AsyncDriver,
    org_id: str,
    user_id: str,
) -> list[dict]:
    """Return the org's subgraphs the *user_id* can access (TASK-209).

    An org's subgraphs are the graphs it ``OWNS``:
    ``(:Organization {org_id})-[:OWNS]->(:Graph:__Platform__)``. This filters
    that set to the graphs on which the caller holds an *active* ``HAS_ROLE``
    edge (``hr.is_active = true``). Soft-deleted graphs
    (``status == 'deactivated'``) are excluded, matching :func:`list_org_graphs`.

    The projection is identical to :func:`list_org_graphs` so both the owner
    and the member paths map cleanly onto ``GraphResponse``.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (:Organization:__Platform__ {org_id: $org_id})
                  -[:OWNS]->(g:Graph:__Platform__)
            WHERE coalesce(g.status, 'active') <> 'deactivated'
              AND EXISTS {
                MATCH (:User:__Platform__ {user_id: $uid})
                      -[hr:HAS_ROLE {graph_id: g.graph_id}]->(:Role:__System__)
                WHERE hr.is_active = true
              }
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
            {"org_id": org_id, "uid": user_id},
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
    logo_url: str | None = None,
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
    if logo_url is not None:
        updates["logo_url"] = logo_url

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
