"""Read-only graph-visualization service (TASK-210).

Builds the node/edge payload for the frontend Graph Explorer's
``GET /api/v1/graphs/{graph_id}/graph-data`` endpoint.

Design — two-phase, fully ``graph_id``-scoped:

  Phase 1  Select the entity (``:__Entity__``) nodes for the graph, apply the
           optional ``node_type`` / ``community_id`` / ``min_degree`` filters,
           compute their entity-to-entity ``degree`` and finest-level
           ``community_id``, count the full match set, then cap to ``limit``.
  Phase 2  Fetch the *induced* relationship set — edges whose BOTH endpoints
           are in the capped node id set. Non-entity edges (``FROM_CHUNK``,
           ``IN_COMMUNITY``, ``PARENT_COMMUNITY``) and cross-graph edges
           (``LINKED_TO``, ``SAME_AS``) are excluded naturally because they
           never connect two ``:__Entity__`` nodes of the same graph.

All Cypher is parameterised. ``graph_id`` appears on every MATCH so a query
can never traverse into another tenant's graph.
"""

from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver

# Marker labels that are NOT a node's display identity. Used both to recognise
# a "real" entity node and to pick its human-readable label.
_RESERVED_LABELS: set[str] = {
    "__Entity__",
    "__KGBuilder__",
    "__Platform__",
    "__Chat__",
    "__Community__",
    "__Rebac__",
    "__System__",
}

# Node properties never emitted to the browser.
#   embedding  — large vector, useless for visualization
#   graph_id   — redundant (the request is already graph-scoped)
#   id         — already surfaced as the top-level node id
_DROPPED_PROPERTIES: set[str] = {"embedding", "graph_id", "id"}

# Hard cap on returned nodes — keeps the payload (and the browser) bounded.
HARD_CAP = 2000
DEFAULT_LIMIT = 500


def _display_label(labels: list[str]) -> str:
    """Pick the first non-reserved label as the node's display label.

    Falls back to ``"Entity"`` when a node carries only reserved markers.
    """
    for label in labels:
        if label not in _RESERVED_LABELS:
            return label
    return "Entity"


def _clean_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Drop embedding / graph_id / id from a node's stored properties."""
    return {k: v for k, v in props.items() if k not in _DROPPED_PROPERTIES}


async def get_graph_data(
    driver: AsyncDriver,
    *,
    graph_id: str,
    limit: int = DEFAULT_LIMIT,
    node_type: str | None = None,
    community_id: str | None = None,
    min_degree: int | None = None,
) -> dict[str, Any]:
    """Return the ``{nodes, edges, truncated}`` visualization payload.

    Args:
        driver: Async Neo4j driver.
        graph_id: Tenant graph to read. Every MATCH is scoped to it.
        limit: Max nodes to return; silently clamped to ``[1, HARD_CAP]``.
        node_type: If set, keep only nodes carrying this label.
        community_id: If set, keep only nodes ``IN_COMMUNITY`` with the active
            ``:__Community__`` of this id in this graph.
        min_degree: If set, keep only nodes whose entity-to-entity degree is
            ``>= min_degree``.

    Returns:
        dict with ``nodes`` (list of node dicts), ``edges`` (list of induced
        edge dicts) and ``truncated`` (bool).
    """
    # Clamp the limit silently — never trust a caller-supplied cap.
    if limit > HARD_CAP:
        limit = HARD_CAP
    if limit < 1:
        limit = 1

    # ---- Phase 1: select + measure + cap the node set --------------------
    #
    # `entity` is a node carrying :__Entity__ and none of the other reserved
    # markers as part of its identity (so :__Community__ nodes are excluded).
    #
    # degree   — count of distinct relationships to ANOTHER :__Entity__ in the
    #            same graph (direction-agnostic; self-loops would count once).
    # community_id — id of an *active* :__Community__ the node is IN_COMMUNITY
    #            with; when several, the lowest `level` (finest) wins.
    #
    # The optional filters are appended as parameterised WHERE clauses. The
    # full match-count is taken BEFORE the LIMIT so `truncated` is accurate.
    node_filters = ["n:__Entity__"]
    for marker in _RESERVED_LABELS - {"__Entity__"}:
        # `marker` is from a fixed internal set — safe to inline as a label.
        node_filters.append(f"NOT n:{marker}")
    if node_type is not None:
        node_filters.append("$node_type IN labels(n)")
    if community_id is not None:
        node_filters.append(
            "EXISTS { "
            "MATCH (n)-[:IN_COMMUNITY {graph_id: $graph_id}]->"
            "(cf:__Community__ {id: $community_id, graph_id: $graph_id}) }"
        )
    where_node = " AND ".join(node_filters)

    min_degree_clause = ""
    if min_degree is not None:
        min_degree_clause = "WHERE degree >= $min_degree"

    select_query = f"""
    MATCH (n {{graph_id: $graph_id}})
    WHERE {where_node}
    WITH n,
         COUNT {{
           MATCH (n)-[r {{graph_id: $graph_id}}]-(m:__Entity__ {{graph_id: $graph_id}})
           RETURN DISTINCT r
         }} AS degree
    {min_degree_clause}
    WITH n, degree
    OPTIONAL MATCH (n)-[ic:IN_COMMUNITY {{graph_id: $graph_id}}]->
                   (c:__Community__ {{graph_id: $graph_id, status: 'active'}})
    WITH n, degree, c
    ORDER BY coalesce(ic.level, 2147483647) ASC
    WITH n, degree, collect(c.id)[0] AS community_id
    WITH collect({{
        id: n.id,
        labels: labels(n),
        type: n.type,
        degree: degree,
        community_id: community_id,
        properties: properties(n)
    }}) AS all_nodes
    RETURN size(all_nodes) AS match_count,
           all_nodes[0..$limit] AS nodes
    """

    params: dict[str, Any] = {"graph_id": graph_id, "limit": limit}
    if node_type is not None:
        params["node_type"] = node_type
    if community_id is not None:
        params["community_id"] = community_id
    if min_degree is not None:
        params["min_degree"] = min_degree

    async with driver.session() as session:
        result = await session.run(select_query, params)
        row = await result.single()

    match_count = (row["match_count"] if row else 0) or 0
    raw_nodes = (row["nodes"] if row else []) or []

    nodes: list[dict[str, Any]] = []
    node_ids: list[str] = []
    for rn in raw_nodes:
        nid = rn["id"]
        node_ids.append(nid)
        nodes.append(
            {
                "id": nid,
                "label": _display_label(rn["labels"] or []),
                "type": rn.get("type"),
                "community_id": rn.get("community_id"),
                "degree": rn.get("degree") or 0,
                "properties": _clean_properties(rn.get("properties") or {}),
            }
        )

    truncated = match_count > len(nodes)

    # ---- Phase 2: induced edges among the capped node set ----------------
    edges: list[dict[str, Any]] = []
    if node_ids:
        edge_query = """
        MATCH (a:__Entity__ {graph_id: $graph_id})
              -[r {graph_id: $graph_id}]->
              (b:__Entity__ {graph_id: $graph_id})
        WHERE a.id IN $node_ids AND b.id IN $node_ids
        RETURN elementId(r) AS id,
               a.id AS source,
               b.id AS target,
               type(r) AS type,
               toFloat(coalesce(r.count, r.score, 1.0)) AS weight
        """
        async with driver.session() as session:
            result = await session.run(
                edge_query, {"graph_id": graph_id, "node_ids": node_ids}
            )
            edge_rows = await result.data()

        for er in edge_rows:
            edges.append(
                {
                    "id": er["id"],
                    "source": er["source"],
                    "target": er["target"],
                    "type": er["type"],
                    "weight": float(er["weight"]) if er["weight"] is not None else 1.0,
                }
            )

    return {"nodes": nodes, "edges": edges, "truncated": truncated}
