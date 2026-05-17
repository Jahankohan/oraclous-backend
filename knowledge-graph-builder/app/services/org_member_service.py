"""Organization Member service (TASK-203 Part 1).

Lets an Organization owner register **members** at the org level and grant
each member access to one / several / all of the org's subgraphs.

Two distinct role systems are in play — do not conflate them:

  - **Org roles** (``owner|admin|member``, ADR-021 §2) live on the
    ``(:User)-[:BELONGS_TO {org_role}]->(:Organization)`` edge. They describe
    standing within the organization itself.
  - **Subgraph ReBAC roles** (``owner|admin|editor|viewer|restricted_viewer``)
    live on the existing ``(:User)-[:HAS_ROLE]->(:Role)`` edges per graph and
    are managed exclusively by :mod:`app.services.rebac_service`. This module
    never reimplements ReBAC — it delegates to ``rebac_service``.

An organization's subgraphs are the graphs it ``OWNS``:
``(:Organization {org_id})-[:OWNS]->(:Graph:__Platform__)``.

This module is purely additive: it does not modify existing per-graph member
endpoints or ReBAC code. All Cypher is parameterized.
"""

from __future__ import annotations

from neo4j import AsyncDriver

from app.core.logging import get_logger
from app.services.rebac_service import rebac_service

logger = get_logger(__name__)

# ADR-021 §2 — the three org-level roles, distinct from ReBAC subgraph roles.
_VALID_ORG_ROLES: frozenset[str] = frozenset({"owner", "admin", "member"})

# The 5 per-subgraph ReBAC roles accepted by rebac_service.grant_role.
_VALID_SUBGRAPH_ROLES: frozenset[str] = frozenset(
    {"owner", "admin", "editor", "viewer", "restricted_viewer"}
)


async def _owned_graph_ids(driver: AsyncDriver, org_id: str) -> list[str]:
    """Return the graph_id of every :Graph the organization OWNS.

    Soft-deleted graphs (``status == 'deactivated'``) are excluded so members
    are never granted onto graphs that have been retired.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (:Organization:__Platform__ {org_id: $org_id})
                  -[:OWNS]->(g:Graph:__Platform__)
            WHERE coalesce(g.status, 'active') <> 'deactivated'
            RETURN g.graph_id AS graph_id
            """,
            {"org_id": org_id},
        )
        return [record["graph_id"] async for record in result]


async def add_member(
    driver: AsyncDriver,
    org_id: str,
    user_id: str,
    org_role: str,
    email: str | None = None,
) -> dict:
    """Register *user_id* as a member of *org_id* with the given *org_role*.

    Idempotent: re-adding an existing member updates the ``org_role`` in place.
    The :User node is MERGE-d (created if absent), matching the convention in
    :func:`organization_service.create_organization`.

    Raises ``ValueError`` if *org_role* is not one of ``owner|admin|member``.
    """
    if org_role not in _VALID_ORG_ROLES:
        raise ValueError(
            f"Invalid org_role {org_role!r} — must be one of {sorted(_VALID_ORG_ROLES)}"
        )

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (o:Organization:__Platform__ {org_id: $org_id})
            MERGE (u:User:__Platform__ {
                user_id: $user_id, graph_id: "__system__"
            })
            ON CREATE SET u.email = $email
            ON MATCH SET u.email = CASE
                WHEN $email IS NOT NULL THEN $email ELSE u.email
            END
            MERGE (u)-[r:BELONGS_TO]->(o)
            ON CREATE SET r.org_role = $org_role, r.since = datetime()
            ON MATCH SET r.org_role = $org_role
            RETURN u.user_id AS user_id,
                   u.email   AS email,
                   r.org_role AS org_role,
                   r.since    AS since
            """,
            {
                "org_id": org_id,
                "user_id": user_id,
                "org_role": org_role,
                "email": email,
            },
        )
        record = await result.single()

    if record is None:
        # The org node does not exist — caller is expected to have verified
        # ownership first, so this is a defensive guard.
        raise ValueError(f"Organization {org_id!r} not found")

    return {
        "user_id": record["user_id"],
        "email": record["email"],
        "org_role": record["org_role"],
        "since": str(record["since"]) if record["since"] is not None else None,
    }


async def list_members(driver: AsyncDriver, org_id: str) -> list[dict]:
    """Return every BELONGS_TO member of *org_id*.

    Each member dict carries ``user_id``, ``email``, ``org_role``, ``since``,
    and ``subgraph_grants`` — a list of ``{graph_id, role}`` derived from the
    member's active ``HAS_ROLE`` edges on graphs the org ``OWNS``.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (u:User:__Platform__)-[r:BELONGS_TO]->
                  (o:Organization:__Platform__ {org_id: $org_id})
            RETURN u.user_id AS user_id,
                   u.email   AS email,
                   r.org_role AS org_role,
                   r.since    AS since
            ORDER BY r.since
            """,
            {"org_id": org_id},
        )
        members = [
            {
                "user_id": record["user_id"],
                "email": record["email"],
                "org_role": record["org_role"],
                "since": (
                    str(record["since"]) if record["since"] is not None else None
                ),
            }
            async for record in result
        ]

    # Resolve subgraph grants by intersecting each owned graph's active
    # HAS_ROLE members with the org's member set. Reuses ReBAC's own query so
    # the "active grant" semantics (is_active + not expired) stay consistent.
    graph_ids = await _owned_graph_ids(driver, org_id)
    grants_by_user: dict[str, list[dict]] = {m["user_id"]: [] for m in members}
    for graph_id in graph_ids:
        graph_members = await rebac_service.list_graph_members(driver, graph_id)
        for gm in graph_members:
            if gm["user_id"] in grants_by_user:
                grants_by_user[gm["user_id"]].append(
                    {"graph_id": graph_id, "role": gm["role"]}
                )

    for member in members:
        member["subgraph_grants"] = grants_by_user.get(member["user_id"], [])

    return members


async def remove_member(driver: AsyncDriver, org_id: str, user_id: str) -> bool:
    """Remove *user_id*'s membership of *org_id*.

    Refuses (raises ``ValueError``) if the user is the organization's only
    member holding the ``owner`` org-role — an org must always retain at least
    one owner.

    Otherwise: deletes the ``BELONGS_TO`` edge and soft-revokes the user's
    ``HAS_ROLE`` grants on every subgraph the org owns (delegating to
    ``rebac_service.revoke_role`` per (graph, role) pair). Returns True.
    """
    async with driver.session() as session:
        # Determine the member's org_role and the org's total owner count in
        # one query, so the last-owner check is race-free within this read.
        result = await session.run(
            """
            MATCH (u:User:__Platform__ {user_id: $user_id})
                  -[r:BELONGS_TO]->
                  (o:Organization:__Platform__ {org_id: $org_id})
            OPTIONAL MATCH (:User:__Platform__)-[r2:BELONGS_TO]->(o)
            WHERE r2.org_role = "owner"
            RETURN r.org_role AS org_role, count(DISTINCT r2) AS owner_count
            """,
            {"org_id": org_id, "user_id": user_id},
        )
        record = await result.single()

    if record is None:
        raise ValueError(f"User {user_id!r} is not a member of organization {org_id!r}")

    if record["org_role"] == "owner" and record["owner_count"] <= 1:
        raise ValueError(
            "Cannot remove the organization's only owner — assign another owner first"
        )

    # Soft-revoke the member's ReBAC grants on every subgraph the org owns,
    # before deleting the membership edge.
    graph_ids = await _owned_graph_ids(driver, org_id)
    for graph_id in graph_ids:
        graph_members = await rebac_service.list_graph_members(driver, graph_id)
        for gm in graph_members:
            if gm["user_id"] == user_id:
                await rebac_service.revoke_role(driver, graph_id, user_id, gm["role"])

    async with driver.session() as session:
        await session.run(
            """
            MATCH (u:User:__Platform__ {user_id: $user_id})
                  -[r:BELONGS_TO]->
                  (:Organization:__Platform__ {org_id: $org_id})
            DELETE r
            """,
            {"org_id": org_id, "user_id": user_id},
        )

    logger.info("Removed member %s from organization %s", user_id, org_id)
    return True


async def grant_member_subgraphs(
    driver: AsyncDriver,
    org_id: str,
    user_id: str,
    role: str,
    graph_ids: list[str] | str,
    granted_by: str,
) -> list[dict]:
    """Grant *user_id* a ReBAC *role* on the org's subgraphs.

    *role* is a per-subgraph ReBAC role (``owner|admin|editor|viewer|
    restricted_viewer``). *graph_ids* is either an explicit list of graph_id
    strings or the literal string ``"all"`` (every subgraph the org owns).

    When a list is given, each graph is verified to be owned by *org_id*; any
    graph not owned by the org raises ``ValueError`` (fail-closed — no partial
    grants onto graphs outside the org's scope).

    Each target graph is granted via ``rebac_service.grant_role``. Returns the
    list of ``{graph_id, role}`` actually granted.
    """
    if role not in _VALID_SUBGRAPH_ROLES:
        raise ValueError(
            f"Invalid subgraph role {role!r} — must be one of "
            f"{sorted(_VALID_SUBGRAPH_ROLES)}"
        )

    owned = await _owned_graph_ids(driver, org_id)

    if graph_ids == "all":
        targets = owned
    elif isinstance(graph_ids, list):
        owned_set = set(owned)
        not_owned = [g for g in graph_ids if g not in owned_set]
        if not_owned:
            raise ValueError(
                f"Graph(s) {not_owned!r} are not owned by organization {org_id!r}"
            )
        targets = graph_ids
    else:
        raise ValueError("graph_ids must be a list of graph ids or the literal 'all'")

    granted: list[dict] = []
    for graph_id in targets:
        await rebac_service.grant_role(
            driver,
            graph_id=graph_id,
            target_user_id=user_id,
            role_name=role,
            granted_by=granted_by,
        )
        granted.append({"graph_id": graph_id, "role": role})

    logger.info(
        "Granted %s role %r on %d subgraph(s) of org %s",
        user_id,
        role,
        len(granted),
        org_id,
    )
    return granted
