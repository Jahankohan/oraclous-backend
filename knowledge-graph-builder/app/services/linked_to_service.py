"""Cross-subgraph LINKED_TO primitive (TASK-204, ADR-021 §4 Mechanism A).

A ``LINKED_TO`` edge is a **directional, intra-organization** link:

  - Subgraph-level:
    ``(:Graph:__Platform__ {graph_id:$src})-[:LINKED_TO {min_role, created_by,
    created_at}]->(:Graph:__Platform__ {graph_id:$tgt})``
  - Entity-level:
    ``(:__Entity__ {graph_id:$src, id:$src_entity})-[:LINKED_TO {...}]->
    (:__Entity__ {graph_id:$tgt, id:$tgt_entity})``

``LINKED_TO`` is a brand-new edge type. It is deliberately **separate** from
federation's ``SAME_AS`` (:mod:`app.services.federation_service`) — that is a
different concern and is never touched or reused here.

Invariants enforced at creation (fail-closed — raise ``ValueError``):

  - Both graphs exist and share the same non-null ``org_id`` (intra-org only).
  - ``src != tgt`` (no self-link); for entity links the two graphs must differ.
  - ``min_role`` is one of the 5 ReBAC role names.
  - For entity links, both entities exist in their stated graphs.

Visibility / ``min_role`` rule (ADR-021 §4 — **source** subgraph): a principal
may see and traverse a ``LINKED_TO`` link only if its ReBAC role on the
**source** subgraph is at or above the link's ``min_role`` — i.e.
``user_role_index(source_graph) <= _SYSTEM_ROLES.index(link.min_role)``
(lower index = more privileged). A principal with no role, or a lower role,
has the link hidden from list results.

The ``:__Entity__`` per-graph key property is ``id`` — verified against the
entity MERGE in :mod:`app.services.structured_ingest_service`
(``MERGE (e:`{label}`:__Entity__ {id: $id, graph_id: $gid})``).

All Cypher is parameterized. This module is purely additive.
"""

from __future__ import annotations

from neo4j import AsyncDriver

from app.core.logging import get_logger
from app.services.rebac_service import _SYSTEM_ROLES

logger = get_logger(__name__)

# The 5 ReBAC subgraph roles, most→least privileged. List index = privilege
# level (0 = owner … 4 = restricted_viewer). Re-used from rebac_service so the
# ordering stays the single source of truth.
_VALID_ROLES: frozenset[str] = frozenset(_SYSTEM_ROLES)


# ── Schema initialization ───────────────────────────────────────────────────


async def initialize_schema(driver: AsyncDriver) -> None:
    """Create LINKED_TO lookup indexes. Idempotent (IF NOT EXISTS).

    Mirrors the startup schema-init pattern used by ``rebac_service`` and
    ``service_account_service`` — safe to call on every app startup.
    """
    index_queries = [
        # Relationship property index on the visibility threshold.
        "CREATE INDEX linked_to_min_role IF NOT EXISTS "
        "FOR ()-[l:LINKED_TO]-() ON (l.min_role)",
        # Graph lookups by graph_id are the join key for both edge variants.
        # rebac_service already creates `rebac_graph_id_idx` on (:Graph) —
        # this adds the __Entity__ counterpart so entity-pair MATCH is indexed.
        "CREATE INDEX linked_to_entity_graph_id IF NOT EXISTS "
        "FOR (e:__Entity__) ON (e.graph_id)",
        "CREATE INDEX linked_to_entity_id IF NOT EXISTS FOR (e:__Entity__) ON (e.id)",
    ]
    async with driver.session() as session:
        for q in index_queries:
            await session.run(q)
    logger.info("LINKED_TO Neo4j schema indexes created/verified")


# ── Role helpers ────────────────────────────────────────────────────────────


async def _user_role_level(
    driver: AsyncDriver, user_id: str, graph_id: str
) -> int | None:
    """Return the user's ReBAC role index on *graph_id* (0–4), else ``None``.

    The role is the ``name`` on the active
    ``(:User)-[hr:HAS_ROLE {graph_id}]->(:Role:__System__ {graph_id, name})``
    edge (``hr.is_active = true``). The returned index is into
    ``_SYSTEM_ROLES`` — lower = more privileged. ``None`` means the user holds
    no active role on the graph.

    If the user somehow holds several active roles, the most privileged
    (lowest index) wins.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (u:User:__Platform__ {user_id: $user_id})
                  -[hr:HAS_ROLE {graph_id: $graph_id}]->
                  (r:Role:__System__ {graph_id: $graph_id})
            WHERE hr.is_active = true
            RETURN collect(DISTINCT r.name) AS role_names
            """,
            {"user_id": user_id, "graph_id": graph_id},
        )
        record = await result.single()

    if record is None:
        return None
    names = [n for n in (record["role_names"] or []) if n in _VALID_ROLES]
    if not names:
        return None
    return min(_SYSTEM_ROLES.index(n) for n in names)


# ── Invariant checks ────────────────────────────────────────────────────────


async def _assert_same_org_graphs(
    driver: AsyncDriver, source_graph_id: str, target_graph_id: str
) -> None:
    """Verify both graphs exist and share the same non-null ``org_id``.

    Raises ``ValueError`` (fail-closed) if either graph is missing, if either
    has no ``org_id``, or if the two ``org_id`` values differ.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            OPTIONAL MATCH (s:Graph:__Platform__ {graph_id: $src})
            OPTIONAL MATCH (t:Graph:__Platform__ {graph_id: $tgt})
            RETURN s IS NOT NULL AS src_exists,
                   t IS NOT NULL AS tgt_exists,
                   s.org_id      AS src_org,
                   t.org_id      AS tgt_org
            """,
            {"src": source_graph_id, "tgt": target_graph_id},
        )
        record = await result.single()

    if record is None or not record["src_exists"]:
        raise ValueError(f"Source graph {source_graph_id!r} not found")
    if not record["tgt_exists"]:
        raise ValueError(f"Target graph {target_graph_id!r} not found")

    src_org = record["src_org"]
    tgt_org = record["tgt_org"]
    if src_org is None or tgt_org is None:
        raise ValueError(
            "Both graphs must belong to an organization to be linked "
            "(LINKED_TO is intra-organization only)"
        )
    if src_org != tgt_org:
        raise ValueError(
            "LINKED_TO is intra-organization only — source and target graphs "
            f"belong to different organizations ({src_org!r} != {tgt_org!r})"
        )


def _validate_min_role(min_role: str) -> None:
    """Raise ``ValueError`` unless *min_role* is one of the 5 ReBAC roles."""
    if min_role not in _VALID_ROLES:
        raise ValueError(
            f"Invalid min_role {min_role!r} — must be one of {sorted(_VALID_ROLES)}"
        )


# ── Graph-level links ───────────────────────────────────────────────────────


async def create_graph_link(
    driver: AsyncDriver,
    source_graph_id: str,
    target_graph_id: str,
    min_role: str,
    created_by: str,
) -> dict:
    """Create (or update) a subgraph-level LINKED_TO edge.

    Enforces all invariants, then MERGEs the edge so re-creation is idempotent
    (``ON MATCH SET`` refreshes ``min_role``, ``created_by``, ``created_at``).

    Raises ``ValueError`` on any invariant violation. Returns
    ``{source_graph_id, target_graph_id, min_role, created_by, created_at}``.
    """
    if source_graph_id == target_graph_id:
        raise ValueError("Cannot link a subgraph to itself")
    _validate_min_role(min_role)
    await _assert_same_org_graphs(driver, source_graph_id, target_graph_id)

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Graph:__Platform__ {graph_id: $src})
            MATCH (t:Graph:__Platform__ {graph_id: $tgt})
            MERGE (s)-[l:LINKED_TO]->(t)
            ON CREATE SET l.min_role   = $min_role,
                          l.created_by = $created_by,
                          l.created_at = datetime()
            ON MATCH SET  l.min_role   = $min_role,
                          l.created_by = $created_by,
                          l.created_at = datetime()
            RETURN s.graph_id AS source_graph_id,
                   t.graph_id AS target_graph_id,
                   l.min_role   AS min_role,
                   l.created_by AS created_by,
                   l.created_at AS created_at
            """,
            {
                "src": source_graph_id,
                "tgt": target_graph_id,
                "min_role": min_role,
                "created_by": created_by,
            },
        )
        record = await result.single()

    if record is None:
        # Both graphs were just verified to exist — defensive guard only.
        raise ValueError("Failed to create LINKED_TO edge — graph not found")

    logger.info(
        "Created LINKED_TO %s -> %s (min_role=%s)",
        source_graph_id,
        target_graph_id,
        min_role,
    )
    return {
        "source_graph_id": record["source_graph_id"],
        "target_graph_id": record["target_graph_id"],
        "min_role": record["min_role"],
        "created_by": record["created_by"],
        "created_at": str(record["created_at"]),
    }


async def list_graph_links(
    driver: AsyncDriver, source_graph_id: str, user_id: str
) -> list[dict]:
    """List outbound subgraph LINKED_TO edges from *source_graph_id*.

    Applies the ADR-021 §4 visibility filter: a link is returned only if the
    caller's ReBAC role on the **source** subgraph is at or above the link's
    ``min_role``. A caller with no role on the source graph sees nothing.
    """
    user_level = await _user_role_level(driver, user_id, source_graph_id)
    if user_level is None:
        # No role on the source subgraph — every link is hidden.
        return []

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (s:Graph:__Platform__ {graph_id: $src})
                  -[l:LINKED_TO]->(t:Graph:__Platform__)
            RETURN s.graph_id AS source_graph_id,
                   t.graph_id AS target_graph_id,
                   l.min_role   AS min_role,
                   l.created_by AS created_by,
                   l.created_at AS created_at
            ORDER BY l.created_at
            """,
            {"src": source_graph_id},
        )
        rows = [
            {
                "source_graph_id": r["source_graph_id"],
                "target_graph_id": r["target_graph_id"],
                "min_role": r["min_role"],
                "created_by": r["created_by"],
                "created_at": str(r["created_at"]),
            }
            async for r in result
        ]

    # Visible iff the caller's role index <= the link's min_role index.
    return [
        row
        for row in rows
        if row["min_role"] in _VALID_ROLES
        and user_level <= _SYSTEM_ROLES.index(row["min_role"])
    ]


async def delete_graph_link(
    driver: AsyncDriver, source_graph_id: str, target_graph_id: str
) -> bool:
    """Delete the subgraph-level LINKED_TO edge ``source -> target``.

    Returns ``True`` if an edge was deleted, ``False`` if there was none.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (:Graph:__Platform__ {graph_id: $src})
                  -[l:LINKED_TO]->(:Graph:__Platform__ {graph_id: $tgt})
            DELETE l
            RETURN count(l) AS deleted
            """,
            {"src": source_graph_id, "tgt": target_graph_id},
        )
        record = await result.single()

    deleted = bool(record and record["deleted"] > 0)
    if deleted:
        logger.info("Deleted LINKED_TO %s -> %s", source_graph_id, target_graph_id)
    return deleted


# ── Entity-level links ──────────────────────────────────────────────────────


async def create_entity_link(
    driver: AsyncDriver,
    source_graph_id: str,
    source_entity_id: str,
    target_graph_id: str,
    target_entity_id: str,
    min_role: str,
    created_by: str,
) -> dict:
    """Create (or update) an entity-level LINKED_TO edge.

    The two entities must live in two **different** subgraphs of the same
    organization, and both must already exist. The edge is MERGEd so
    re-creation is idempotent.

    Raises ``ValueError`` on any invariant violation. Returns the link dict.
    """
    if source_graph_id == target_graph_id:
        raise ValueError(
            "Entity links must cross subgraphs — source and target graphs must differ"
        )
    _validate_min_role(min_role)
    await _assert_same_org_graphs(driver, source_graph_id, target_graph_id)

    async with driver.session() as session:
        # Verify both entities exist in their stated graphs, then MERGE.
        result = await session.run(
            """
            OPTIONAL MATCH (s:__Entity__ {graph_id: $src_g, id: $src_e})
            OPTIONAL MATCH (t:__Entity__ {graph_id: $tgt_g, id: $tgt_e})
            WITH s, t
            WHERE s IS NOT NULL AND t IS NOT NULL
            MERGE (s)-[l:LINKED_TO]->(t)
            ON CREATE SET l.min_role   = $min_role,
                          l.created_by = $created_by,
                          l.created_at = datetime()
            ON MATCH SET  l.min_role   = $min_role,
                          l.created_by = $created_by,
                          l.created_at = datetime()
            RETURN s.graph_id AS source_graph_id,
                   s.id       AS source_entity_id,
                   t.graph_id AS target_graph_id,
                   t.id       AS target_entity_id,
                   l.min_role   AS min_role,
                   l.created_by AS created_by,
                   l.created_at AS created_at
            """,
            {
                "src_g": source_graph_id,
                "src_e": source_entity_id,
                "tgt_g": target_graph_id,
                "tgt_e": target_entity_id,
                "min_role": min_role,
                "created_by": created_by,
            },
        )
        record = await result.single()

    if record is None:
        # The WHERE filtered out the row — one of the entities is missing.
        # Determine which, for a precise error message.
        async with driver.session() as session:
            check = await session.run(
                """
                OPTIONAL MATCH (s:__Entity__ {graph_id: $src_g, id: $src_e})
                OPTIONAL MATCH (t:__Entity__ {graph_id: $tgt_g, id: $tgt_e})
                RETURN s IS NOT NULL AS src_exists,
                       t IS NOT NULL AS tgt_exists
                """,
                {
                    "src_g": source_graph_id,
                    "src_e": source_entity_id,
                    "tgt_g": target_graph_id,
                    "tgt_e": target_entity_id,
                },
            )
            cr = await check.single()
        if cr is None or not cr["src_exists"]:
            raise ValueError(
                f"Source entity {source_entity_id!r} not found in graph "
                f"{source_graph_id!r}"
            )
        raise ValueError(
            f"Target entity {target_entity_id!r} not found in graph {target_graph_id!r}"
        )

    logger.info(
        "Created entity LINKED_TO %s/%s -> %s/%s (min_role=%s)",
        source_graph_id,
        source_entity_id,
        target_graph_id,
        target_entity_id,
        min_role,
    )
    return {
        "source_graph_id": record["source_graph_id"],
        "source_entity_id": record["source_entity_id"],
        "target_graph_id": record["target_graph_id"],
        "target_entity_id": record["target_entity_id"],
        "min_role": record["min_role"],
        "created_by": record["created_by"],
        "created_at": str(record["created_at"]),
    }


async def list_entity_links(
    driver: AsyncDriver,
    source_graph_id: str,
    source_entity_id: str,
    user_id: str,
) -> list[dict]:
    """List outbound entity LINKED_TO edges from a source entity.

    Applies the ADR-021 §4 visibility filter against the caller's ReBAC role
    on the **source** subgraph (the graph the source entity belongs to).
    """
    user_level = await _user_role_level(driver, user_id, source_graph_id)
    if user_level is None:
        return []

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (s:__Entity__ {graph_id: $src_g, id: $src_e})
                  -[l:LINKED_TO]->(t:__Entity__)
            RETURN s.graph_id AS source_graph_id,
                   s.id       AS source_entity_id,
                   t.graph_id AS target_graph_id,
                   t.id       AS target_entity_id,
                   l.min_role   AS min_role,
                   l.created_by AS created_by,
                   l.created_at AS created_at
            ORDER BY l.created_at
            """,
            {"src_g": source_graph_id, "src_e": source_entity_id},
        )
        rows = [
            {
                "source_graph_id": r["source_graph_id"],
                "source_entity_id": r["source_entity_id"],
                "target_graph_id": r["target_graph_id"],
                "target_entity_id": r["target_entity_id"],
                "min_role": r["min_role"],
                "created_by": r["created_by"],
                "created_at": str(r["created_at"]),
            }
            async for r in result
        ]

    return [
        row
        for row in rows
        if row["min_role"] in _VALID_ROLES
        and user_level <= _SYSTEM_ROLES.index(row["min_role"])
    ]


async def delete_entity_link(
    driver: AsyncDriver,
    source_graph_id: str,
    source_entity_id: str,
    target_graph_id: str,
    target_entity_id: str,
) -> bool:
    """Delete the entity-level LINKED_TO edge ``source -> target``.

    Returns ``True`` if an edge was deleted, ``False`` if there was none.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (:__Entity__ {graph_id: $src_g, id: $src_e})
                  -[l:LINKED_TO]->
                  (:__Entity__ {graph_id: $tgt_g, id: $tgt_e})
            DELETE l
            RETURN count(l) AS deleted
            """,
            {
                "src_g": source_graph_id,
                "src_e": source_entity_id,
                "tgt_g": target_graph_id,
                "tgt_e": target_entity_id,
            },
        )
        record = await result.single()

    deleted = bool(record and record["deleted"] > 0)
    if deleted:
        logger.info(
            "Deleted entity LINKED_TO %s/%s -> %s/%s",
            source_graph_id,
            source_entity_id,
            target_graph_id,
            target_entity_id,
        )
    return deleted
