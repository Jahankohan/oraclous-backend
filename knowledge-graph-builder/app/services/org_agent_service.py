"""Organization Agent registry service (TASK-203 Part 2).

An :Agent is created bound to a single home graph (its required ``graph_id``).
TASK-203 Part 2 makes an agent **org-aware** and **grantable across subgraphs**:

  - Every agent carries an ``org_id`` (the org that owns its home graph) and
    the org knows its agents via ``(:Organization)-[:HAS_AGENT]->(:Agent)``.
  - An agent can be granted ``CAN_ACCESS`` to one / several / all of the
    organization's subgraphs.

This module mirrors the service-account multi-graph grant model
(:mod:`app.services.service_account_service`): the **same** ``CAN_ACCESS``
edge type and the **same** level vocabulary (``reader|writer|admin``) are used,
so agent grants and SA grants stay consistent.

An organization's subgraphs are the graphs it ``OWNS``:
``(:Organization {org_id})-[:OWNS]->(:Graph:__Platform__)`` — resolved (and
soft-deleted graphs excluded) via
:func:`app.services.org_member_service._owned_graph_ids`.

This module is purely additive: it does not modify existing per-graph agent
CRUD or chat. All Cypher is parameterized. ``check_agent_graph_permission`` is
fail-closed — it returns True only when an explicit allow can be proven.

Scope boundary: this module delivers the grant machinery + the permission
check + the registry only. Fanning agent retrieval across granted subgraphs at
chat time is TASK-205 and is intentionally NOT implemented here.
"""

from __future__ import annotations

from neo4j import AsyncDriver

from app.core.logging import get_logger
from app.services.org_member_service import _owned_graph_ids

logger = get_logger(__name__)

# Agent grant levels — identical vocabulary to ServiceAccount CAN_ACCESS grants.
_VALID_LEVELS: frozenset[str] = frozenset({"reader", "writer", "admin"})

# Total order on the levels: reader < writer < admin.
_LEVEL_ORDER: list[str] = ["reader", "writer", "admin"]

# For a given *required* level, the set of *granted* levels that satisfy it —
# i.e. every level at or above the requirement. A 'writer' grant satisfies a
# 'reader' requirement; a 'reader' grant does NOT satisfy an 'admin'
# requirement. Used by check_agent_graph_permission (fail-closed).
_SATISFYING_LEVELS: dict[str, list[str]] = {
    required: _LEVEL_ORDER[idx:] for idx, required in enumerate(_LEVEL_ORDER)
}


async def _agent_belongs_to_org(
    driver: AsyncDriver, org_id: str, agent_id: str
) -> bool:
    """Return True iff ``(:Organization {org_id})-[:HAS_AGENT]->(:Agent {agent_id})``."""
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (:Organization:__Platform__ {org_id: $org_id})
                  -[:HAS_AGENT]->(a:Agent:__Platform__ {agent_id: $agent_id})
            RETURN a.agent_id AS agent_id
            LIMIT 1
            """,
            {"org_id": org_id, "agent_id": agent_id},
        )
        record = await result.single()
    return record is not None


async def grant_agent_subgraphs(
    driver: AsyncDriver,
    org_id: str,
    agent_id: str,
    level: str,
    graph_ids: list[str] | str,
    granted_by: str,
) -> list[dict]:
    """Grant *agent_id* a ``CAN_ACCESS`` *level* on the org's subgraphs.

    *level* is one of ``reader|writer|admin`` (ValueError otherwise).
    *graph_ids* is either an explicit list of graph_id strings or the literal
    string ``"all"`` (every subgraph the org owns).

    The agent must belong to *org_id* (``HAS_AGENT`` edge) — ValueError if not.
    When a list is given, each graph is verified to be owned by *org_id*; any
    graph not owned by the org raises ValueError (fail-closed — no partial
    grants onto graphs outside the org's scope).

    Each target graph gets a ``CAN_ACCESS`` edge with ``source='explicit'``.
    Returns the list of ``{graph_id, level}`` actually granted.
    """
    if level not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid level {level!r} — must be one of {sorted(_VALID_LEVELS)}"
        )

    if not await _agent_belongs_to_org(driver, org_id, agent_id):
        raise ValueError(
            f"Agent {agent_id!r} does not belong to organization {org_id!r}"
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
    async with driver.session() as session:
        for graph_id in targets:
            await session.run(
                """
                MATCH (a:Agent:__Platform__ {agent_id: $agent_id})
                MATCH (g:Graph:__Platform__ {graph_id: $graph_id})
                MERGE (a)-[r:CAN_ACCESS]->(g)
                ON CREATE SET
                    r.level      = $level,
                    r.granted_by = $granted_by,
                    r.granted_at = datetime(),
                    r.source     = 'explicit'
                ON MATCH SET
                    r.level      = $level,
                    r.granted_by = $granted_by,
                    r.granted_at = datetime(),
                    r.source     = 'explicit'
                """,
                {
                    "agent_id": agent_id,
                    "graph_id": graph_id,
                    "level": level,
                    "granted_by": granted_by,
                },
            )
            granted.append({"graph_id": graph_id, "level": level})

    logger.info(
        "Granted agent %s level %r on %d subgraph(s) of org %s",
        agent_id,
        level,
        len(granted),
        org_id,
    )
    return granted


async def list_agent_grants(
    driver: AsyncDriver, org_id: str, agent_id: str
) -> list[dict]:
    """Return the agent's ``CAN_ACCESS`` edges as ``{graph_id, level, granted_at}``.

    The agent must belong to *org_id* — ValueError otherwise. Only the
    explicitly-granted subgraphs are listed here; the agent's own home graph is
    not a ``CAN_ACCESS`` edge and is intentionally excluded.
    """
    if not await _agent_belongs_to_org(driver, org_id, agent_id):
        raise ValueError(
            f"Agent {agent_id!r} does not belong to organization {org_id!r}"
        )

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Agent:__Platform__ {agent_id: $agent_id})
                  -[r:CAN_ACCESS]->(g:Graph:__Platform__)
            RETURN g.graph_id          AS graph_id,
                   r.level             AS level,
                   toString(r.granted_at) AS granted_at
            ORDER BY r.granted_at DESC
            """,
            {"agent_id": agent_id},
        )
        return [
            {
                "graph_id": record["graph_id"],
                "level": record["level"],
                "granted_at": record["granted_at"],
            }
            async for record in result
        ]


async def revoke_agent_grant(
    driver: AsyncDriver, org_id: str, agent_id: str, graph_id: str
) -> bool:
    """Delete the agent's ``CAN_ACCESS`` edge to *graph_id*.

    The agent must belong to *org_id* — ValueError otherwise. Returns True if
    an edge was deleted, False if there was no such grant.
    """
    if not await _agent_belongs_to_org(driver, org_id, agent_id):
        raise ValueError(
            f"Agent {agent_id!r} does not belong to organization {org_id!r}"
        )

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Agent:__Platform__ {agent_id: $agent_id})
                  -[r:CAN_ACCESS]->(:Graph:__Platform__ {graph_id: $graph_id})
            DELETE r
            RETURN count(r) AS deleted_count
            """,
            {"agent_id": agent_id, "graph_id": graph_id},
        )
        record = await result.single()
    return bool(record and record["deleted_count"] > 0)


async def check_agent_graph_permission(
    driver: AsyncDriver,
    agent_id: str,
    graph_id: str,
    required_level: str,
) -> bool:
    """Fail-closed check: may *agent_id* act on *graph_id* at *required_level*?

    Returns True iff either:
      1. *graph_id* is the agent's home graph (``a.graph_id``) — the agent
         always has full access to its home graph; or
      2. a ``CAN_ACCESS`` edge from the agent to that graph exists whose
         ``level`` is at or above *required_level* (order: reader < writer <
         admin).

    Returns False in every other case — including an unknown agent, an unknown
    graph, an unknown *required_level*, or a missing/insufficient grant.
    """
    permitted_levels = _SATISFYING_LEVELS.get(required_level)
    if permitted_levels is None:
        # Unknown required_level — deny by default (fail-closed).
        return False

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Agent:__Platform__ {agent_id: $agent_id})
            WITH a, (a.graph_id = $graph_id) AS is_home
            OPTIONAL MATCH (a)-[r:CAN_ACCESS]->(:Graph:__Platform__ {graph_id: $graph_id})
            WITH is_home, collect(r.level) AS grant_levels
            RETURN is_home AS is_home,
                   any(lvl IN grant_levels WHERE lvl IN $permitted_levels)
                       AS has_grant
            """,
            {
                "agent_id": agent_id,
                "graph_id": graph_id,
                "permitted_levels": permitted_levels,
            },
        )
        record = await result.single()

    if record is None:
        # Agent does not exist — deny.
        return False
    return bool(record["is_home"]) or bool(record["has_grant"])


async def list_org_agents(driver: AsyncDriver, org_id: str) -> list[dict]:
    """Return every active :Agent owned by *org_id*.

    An agent belongs to the org when its ``org_id`` property matches. Soft-
    deleted agents (``deactivated_at`` set) are excluded. Each dict carries
    ``agent_id``, ``org_id``, ``graph_id`` (the home graph), ``name``,
    ``description`` and ``deactivated_at`` (always None here, kept for shape
    parity with the agent CRUD projection).
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Agent:__Platform__ {org_id: $org_id})
            WHERE a.deactivated_at IS NULL
            RETURN a.agent_id     AS agent_id,
                   a.org_id       AS org_id,
                   a.graph_id     AS graph_id,
                   a.name         AS name,
                   a.description  AS description
            ORDER BY a.created_at DESC
            """,
            {"org_id": org_id},
        )
        return [
            {
                "agent_id": record["agent_id"],
                "org_id": record["org_id"],
                "graph_id": record["graph_id"],
                "name": record["name"],
                "description": record["description"] or "",
                "deactivated_at": None,
            }
            async for record in result
        ]
