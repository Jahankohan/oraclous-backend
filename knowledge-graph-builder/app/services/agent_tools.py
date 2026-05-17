"""Seven graph-algorithm tools available to graph-native agents (TASK-033 / STORY-020).

Each tool is an async method on AgentToolkit. Instantiate with a driver, the
agent's allowed_tools list, and an optional embedder for graph_search.

ToolNotPermittedError is raised at the top of every method, before any I/O,
so the check cannot be bypassed by the calling code.
"""

import inspect
from typing import Any

from neo4j import AsyncDriver

from app.core.logging import get_logger
from app.schemas.agent_schemas import NodeResult, PathResult

logger = get_logger(__name__)

_MAX_DEPTH = 20  # safety cap on variable-length traversals


class ToolNotPermittedError(Exception):
    """Raised when an agent calls a tool not in its allowlist."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' is not permitted for this agent")
        self.tool_name = tool_name


def _as_graph_ids(graph_id: str | list[str]) -> list[str]:
    """Normalize a tool's first argument to a non-empty list of graph ids.

    TASK-205: agent tools used to take a single ``graph_id: str``. To let a
    query span LINKED_TO subgraphs the executor now passes the effective
    graph-id set (source graph first, then visibility-checked linked
    targets). For full backward compatibility a plain ``str`` is still
    accepted and wrapped into a one-element list — and a one-element list
    produces Cypher results identical to the pre-TASK-205 ``= $gid`` form
    (``IN ['x']`` is equivalent to ``= 'x'``).

    The result is order-preserving and de-duplicated (source graph stays
    first). Raises ``ValueError`` on an empty set so tenant isolation
    fails closed — a query is never run without a graph_id filter.
    """
    if isinstance(graph_id, str):
        ids = [graph_id]
    else:
        ids = list(graph_id)
    seen: set[str] = set()
    out: list[str] = []
    for gid in ids:
        if gid and gid not in seen:
            seen.add(gid)
            out.append(gid)
    if not out:
        raise ValueError("graph_id set must contain at least one graph id")
    return out


def _node_to_result(props: dict[str, Any]) -> NodeResult:
    _skip = {"id", "node_id", "qualified_name", "label", "name", "embedding"}
    return NodeResult(
        id=str(props.get("id") or props.get("node_id") or ""),
        qualified_name=props.get("qualified_name"),
        label=str(props.get("label") or props.get("name") or ""),
        properties={k: v for k, v in props.items() if k not in _skip},
    )


def _safe_label(label: str) -> str:
    """Validate a Neo4j node label used as an f-string literal in Cypher."""
    if not label.replace("_", "").isalnum():
        raise ValueError(
            f"Invalid node label {label!r}: only alphanumeric characters and underscores allowed"
        )
    return label


class AgentToolkit:
    """Executes graph algorithm tools scoped to a single agent's allowlist.

    Parameters
    ----------
    driver:
        Async Neo4j driver.
    allowed_tools:
        Tool names this agent is permitted to call (from the :Agent node).
    embedder:
        Object with an ``embed_query(text)`` method (sync or async) used by
        ``graph_search``. May be None if graph_search is not in allowed_tools.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        allowed_tools: list[str],
        embedder: Any = None,
    ) -> None:
        self._driver = driver
        self._allowed = frozenset(allowed_tools)
        self._embedder = embedder

    # ── Internal ──────────────────────────────────────────────────────────────

    def _require(self, tool_name: str) -> None:
        if tool_name not in self._allowed:
            raise ToolNotPermittedError(tool_name)

    async def _embed(self, text: str) -> list[float]:
        result = self._embedder.embed_query(text)
        return await result if inspect.isawaitable(result) else result

    # ── Tools ─────────────────────────────────────────────────────────────────

    async def graph_search(
        self, graph_id: str | list[str], query: str, max_results: int = 10
    ) -> list[NodeResult]:
        """Semantic similarity search over the graph's embedding space.

        TASK-205: ``graph_id`` may be a single id (legacy) or the effective
        graph-id set spanning LINKED_TO subgraphs. A one-element set is
        byte-identical to the pre-TASK-205 ``= $gid`` query.
        """
        self._require("graph_search")
        gids = _as_graph_ids(graph_id)
        embedding = await self._embed(query)
        # Tenant-scoped cosine: filter to the graph(s) FIRST, then score
        # within. The global `db.index.vector.queryNodes` returns a
        # database-wide top-K — on a multi-graph database the largest graph
        # fills every slot and a smaller graph's nodes never survive the
        # `graph_id` post-filter (cross-tenant starvation: a graph that is
        # not the biggest in the DB retrieves nothing). A full cosine scan
        # over one tenant's chunks is exact and fast at this scale.
        result = await self._driver.execute_query(
            """
            MATCH (node:Chunk)
            WHERE node.graph_id IN $gids AND node.embedding IS NOT NULL
            WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score
            RETURN node, score
            ORDER BY score DESC
            LIMIT $max_results
            """,
            {"gids": gids, "max_results": max_results, "embedding": embedding},
        )
        return [_node_to_result(dict(rec["node"])) for rec in result.records]

    async def community_members(
        self, graph_id: str | list[str], community_id: str, max_results: int = 50
    ) -> list[NodeResult]:
        """Return all member nodes of a community across any registered kind.

        Each registered kind in ``COMMUNITY_KINDS`` is probed in order until
        a match for ``community_id`` is found. This avoids forcing the
        calling LLM to know whether ``community_id`` points at an entity
        community or a chunk community.

        TASK-205: ``graph_id`` may be the effective graph-id set; both the
        member node and its community must live in that set.
        """
        self._require("community_members")
        gids = _as_graph_ids(graph_id)
        # Imported lazily so the toolkit can still be used in tests that
        # don't have the full app package wired up.
        from app.schemas.community_kinds import all_kinds

        for spec in all_kinds():
            # Labels, relationship types, and property names come from the
            # compile-time registry — never from user input — so f-string
            # interpolation here is safe. Identifiers wrapped in backticks
            # for defense-in-depth.
            query = (
                f"MATCH (n:`{spec.member_label}`)"
                f"-[:`{spec.member_rel}`]->"
                f"(c:`{spec.community_label}` "
                f"{{`{spec.id_property}`: $cid}}) "
                "WHERE n.graph_id IN $gids AND c.graph_id IN $gids "
                "RETURN n "
                "LIMIT $max_results"
            )
            result = await self._driver.execute_query(
                query,
                {"gids": gids, "cid": community_id, "max_results": max_results},
            )
            if result.records:
                return [_node_to_result(dict(rec["n"])) for rec in result.records]

        return []

    async def neighbors(
        self,
        graph_id: str | list[str],
        node_id: str,
        edge_type: str | None = None,
        depth: int = 1,
    ) -> list[NodeResult]:
        """BFS from node_id up to `depth` hops, optionally filtered by relationship type.

        TASK-205 — single-graph by design. A bounded variable-length
        traversal is scoped to the **source** graph only (the first id in
        the effective set). Knowledge edges (``FROM_CHUNK`` etc.) never
        span subgraphs, so widening the node filter to the linked set
        would not reach linked data anyway while changing path
        cardinality semantics. The source graph is used; linked targets
        are intentionally ignored for this traversal tool.
        """
        self._require("neighbors")
        graph_id = _as_graph_ids(graph_id)[0]
        depth = min(max(1, depth), _MAX_DEPTH)
        # depth embedded as f-string — Neo4j 5.x does not allow range upper bounds as params
        edge_filter = (
            "AND ALL(rel IN r WHERE TYPE(rel) = $edge_type)" if edge_type else ""
        )
        query = f"""
        MATCH (n {{graph_id: $gid, id: $nid}})-[r*1..{depth}]-(m)
        WHERE m.graph_id = $gid {edge_filter}
        RETURN DISTINCT m
        """
        params: dict[str, Any] = {"gid": graph_id, "nid": node_id}
        if edge_type:
            params["edge_type"] = edge_type
        result = await self._driver.execute_query(query, params)
        return [_node_to_result(dict(rec["m"])) for rec in result.records]

    async def degree_centrality(
        self, graph_id: str | list[str], node_label: str, top_n: int = 10
    ) -> list[NodeResult]:
        """Return top_n most-connected nodes of the given label (Cypher COUNT, not GDS).

        TASK-205: spans the effective graph-id set; a one-element set is
        identical to the pre-TASK-205 query.
        """
        self._require("degree_centrality")
        gids = _as_graph_ids(graph_id)
        label = _safe_label(node_label)
        # label cannot be a Cypher parameter; validated above
        query = f"""
        MATCH (n:{label})-[r]-()
        WHERE n.graph_id IN $gids
        RETURN n, count(r) AS degree
        ORDER BY degree DESC
        LIMIT $top_n
        """
        result = await self._driver.execute_query(query, {"gids": gids, "top_n": top_n})
        return [_node_to_result(dict(rec["n"])) for rec in result.records]

    async def shortest_path(
        self, graph_id: str | list[str], from_qname: str, to_qname: str
    ) -> PathResult | None:
        """Return shortest undirected path between two nodes by qualified_name.

        TASK-205 — single-graph by design. Every node on the path must
        share one ``graph_id``; knowledge edges do not cross subgraphs,
        so the traversal is scoped to the **source** graph (first id of
        the effective set). Linked targets are intentionally ignored.
        """
        self._require("shortest_path")
        graph_id = _as_graph_ids(graph_id)[0]
        result = await self._driver.execute_query(
            """
            MATCH p = shortestPath(
                (a {graph_id: $gid})-[*]-(b {graph_id: $gid})
            )
            WHERE a.qualified_name = $from AND b.qualified_name = $to
              AND ALL(n IN nodes(p) WHERE n.graph_id = $gid)
            RETURN p, length(p) AS hop_count
            """,
            {"gid": graph_id, "from": from_qname, "to": to_qname},
        )
        if not result.records:
            return None
        rec = result.records[0]
        path = rec["p"]
        nodes = [_node_to_result(dict(n)) for n in path.nodes]
        return PathResult(nodes=nodes, hop_count=rec["hop_count"])

    async def taint_trace(
        self, graph_id: str | list[str], source_qname: str, depth: int = 10
    ) -> list[NodeResult]:
        """Follow FLOWS_TO edges from source_qname (valid for code knowledge graphs).

        TASK-205 — single-graph by design. ``FLOWS_TO`` is an intra-graph
        code edge; the traversal is scoped to the **source** graph (first
        id of the effective set). Linked targets are intentionally ignored.
        """
        self._require("taint_trace")
        graph_id = _as_graph_ids(graph_id)[0]
        depth = min(max(1, depth), _MAX_DEPTH)
        # depth embedded as f-string — Neo4j 5.x does not allow range upper bounds as params
        query = f"""
        MATCH (src {{graph_id: $gid, qualified_name: $qname}})-[:FLOWS_TO*1..{depth}]->(sink)
        WHERE sink.graph_id = $gid
        RETURN DISTINCT sink
        """
        result = await self._driver.execute_query(
            query, {"gid": graph_id, "qname": source_qname}
        )
        return [_node_to_result(dict(rec["sink"])) for rec in result.records]

    async def cypher_query(
        self,
        graph_id: str | list[str],
        cypher: str,
        max_results: int = 25,
    ) -> list[NodeResult]:
        """Execute a read-only Cypher query scoped to this graph.

        The agent's LLM generates Cypher (text2cypher pattern). Safety rules:
          - Block any write operation (CREATE, DELETE, SET, REMOVE, MERGE, DROP).
          - Block APOC procedures that can mutate.
          - Require the query to constrain by ``graph_id`` (tenant isolation).
          - Cap result size at ``max_results`` via injected LIMIT clause if absent.

        TASK-205 — single-graph by design. The LLM generates the Cypher
        text and binds ``$graph_id`` as a scalar; spanning linked
        subgraphs would require re-prompting the model to emit
        ``IN``-form queries. The query stays scoped to the **source**
        graph (first id of the effective set).

        Returns the first node-valued column from each record.
        """
        self._require("cypher_query")
        graph_id = _as_graph_ids(graph_id)[0]
        if not isinstance(cypher, str) or not cypher.strip():
            raise ValueError("cypher must be a non-empty string")
        # Strip leading EXPLAIN/PROFILE for safety check, but reject EXPLAIN/PROFILE writes.
        normalized = " ".join(cypher.split()).upper()
        blocked = (
            " CREATE ",
            " DELETE ",
            " SET ",
            " REMOVE ",
            " MERGE ",
            " DROP ",
            "DETACH DELETE",
            "CALL APOC.PERIODIC",
            "CALL APOC.REFACTOR",
            "CALL APOC.CREATE",
            "CALL APOC.MERGE",
            "CALL DBMS.",
        )
        padded = f" {normalized} "
        for kw in blocked:
            if kw in padded:
                raise ValueError(f"Cypher write/admin op rejected: {kw.strip()!r}")
        if "GRAPH_ID" not in normalized:
            raise ValueError(
                "Cypher must filter by graph_id for tenant isolation. "
                "Use $graph_id as a parameter, e.g. WHERE n.graph_id = $graph_id"
            )
        max_results = min(max(1, int(max_results)), 100)
        if " LIMIT " not in padded:
            cypher = cypher.rstrip().rstrip(";") + f"\nLIMIT {max_results}"
        result = await self._driver.execute_query(
            cypher, {"graph_id": graph_id, "gid": graph_id, "graphid": graph_id}
        )
        from neo4j.graph import Node as _NeoNode  # local import to avoid cycle

        nodes: list[NodeResult] = []
        for rec in result.records:
            row_keys = list(rec.keys())
            # Try to extract a Node-valued column first.
            row_node = None
            for key in row_keys:
                val = rec[key]
                if isinstance(val, _NeoNode):
                    row_node = _node_to_result(dict(val))
                    break
            if row_node is not None:
                nodes.append(row_node)
                continue
            # No node in this row — produce a synthetic NodeResult that surfaces
            # the projected columns (counts, names, aggregates, etc.) so the
            # agent can read them. label = first column as string.
            props = {k: rec[k] for k in row_keys}
            display_parts = [f"{k}={rec[k]}" for k in row_keys]
            label = "; ".join(display_parts)[:200]
            nodes.append(_node_to_result({"label": label, **props}))
        return nodes[:max_results]

    async def temporal_slice(
        self,
        graph_id: str | list[str],
        node_label: str,
        at_time: int,
        max_results: int = 50,
    ) -> list[NodeResult]:
        """Return nodes valid at a given point in time (bitemporal valid_from/valid_to).

        TASK-205: spans the effective graph-id set; a one-element set is
        identical to the pre-TASK-205 query.
        """
        self._require("temporal_slice")
        gids = _as_graph_ids(graph_id)
        label = _safe_label(node_label)
        # label cannot be a Cypher parameter; validated above
        query = f"""
        MATCH (n:{label})
        WHERE n.graph_id IN $gids
          AND n.valid_from <= $at_time
          AND (n.valid_to IS NULL OR n.valid_to >= $at_time)
        RETURN n
        LIMIT $max_results
        """
        result = await self._driver.execute_query(
            query, {"gids": gids, "at_time": at_time, "max_results": max_results}
        )
        return [_node_to_result(dict(rec["n"])) for rec in result.records]

    # ── STORY-4d: community discovery + description tools ────────────────────

    async def find_communities(
        self,
        graph_id: str | list[str],
        query: str,
        kind: str | None = None,
        top_k: int = 5,
    ) -> list[NodeResult]:
        """Find communities related to a query via vector search over summaries.

        TASK-205: spans the effective graph-id set; a one-element set is
        identical to the pre-TASK-205 query.

        ``kind=None`` searches every registered kind that has a populated
        vector index, then merges results by score. ``kind="chunk"`` or
        ``kind="entity"`` restricts to one kind.

        Each result's ``properties`` carries ``kind``, ``score``,
        ``summary``, ``keywords`` (chunk only), ``excerpt`` (chunk only),
        ``size``. The ``label`` field surfaces the summary preview so
        agent prompts can read the result in a glance.
        """
        self._require("find_communities")
        gids = _as_graph_ids(graph_id)
        from app.schemas.community_kinds import all_kinds, get_kind

        if kind is not None:
            specs = [get_kind(kind)]
        else:
            specs = list(all_kinds())

        embedding = await self._embed(query)
        per_index_k = max(top_k, 10) if len(specs) > 1 else top_k

        merged: list[tuple[float, dict[str, Any]]] = []
        for spec in specs:
            # Index name, label, and property names come from the registry —
            # compile-time constants, safe for f-string interpolation. The
            # graph_id filter is parameterized so tenant isolation holds.
            cypher = (
                f"CALL db.index.vector.queryNodes('{spec.index_name}', "
                f"$top_k, $embedding) YIELD node, score "
                f"WHERE node.graph_id IN $gids AND node.summary IS NOT NULL "
                f"RETURN node.`{spec.id_property}` AS community_id, "
                f"node.summary AS summary, "
                f"node.`{spec.size_property}` AS size, "
                "node.summary_keywords AS keywords, "
                "node.summary_excerpt AS excerpt, "
                "score "
                "ORDER BY score DESC"
            )
            try:
                result = await self._driver.execute_query(
                    cypher,
                    {
                        "gids": gids,
                        "top_k": per_index_k,
                        "embedding": embedding,
                    },
                )
            except Exception as exc:
                # An index missing on one kind shouldn't break the whole
                # query — log and try the next kind. The LLM-facing tool
                # result is still useful if at least one kind responded.
                logger.warning(
                    "find_communities index %s failed: %s", spec.index_name, exc
                )
                continue

            for rec in result.records:
                # Parse keywords JSON for chunk; entity returns NULL.
                kw_raw = rec.get("keywords")
                keywords: list[str] | None = None
                if kw_raw:
                    try:
                        import json as _json

                        parsed = _json.loads(kw_raw)
                        if isinstance(parsed, list):
                            keywords = parsed
                    except (ValueError, TypeError):
                        keywords = None
                community_id = rec["community_id"]
                merged.append(
                    (
                        rec["score"],
                        {
                            "id": community_id,
                            "label": (rec.get("summary") or "")[:160],
                            "qualified_name": None,
                            "properties": {
                                "kind": spec.kind,
                                "score": rec["score"],
                                "summary": rec.get("summary"),
                                "keywords": keywords,
                                "excerpt": rec.get("excerpt"),
                                "size": rec.get("size"),
                            },
                        },
                    )
                )

        # Highest score first; cap at top_k overall when multi-kind.
        merged.sort(key=lambda x: x[0], reverse=True)
        return [
            NodeResult(
                id=p["id"],
                qualified_name=p["qualified_name"],
                label=p["label"],
                properties=p["properties"],
            )
            for _score, p in merged[:top_k]
        ]

    async def describe_community(
        self,
        graph_id: str | list[str],
        community_id: str,
        kind: str | None = None,
    ) -> list[NodeResult]:
        """Return metadata for one community: summary, keywords, excerpt,
        size, plus up to 5 sample member names.

        TASK-205: spans the effective graph-id set; a one-element set is
        identical to the pre-TASK-205 query.

        Auto-detects kind by probing the registry (entity first → one
        round trip on the common case) when ``kind`` is None. The agent
        can also pass ``kind`` explicitly to skip the probe.

        Returns a single-element ``list[NodeResult]`` for shape
        consistency with the other agent tools. The result's
        ``properties`` carry every relevant field; the ``label`` is a
        summary preview.
        """
        self._require("describe_community")
        gids = _as_graph_ids(graph_id)
        from app.schemas.community_kinds import all_kinds, get_kind

        if kind is not None:
            specs = [get_kind(kind)]
        else:
            specs = list(all_kinds())

        for spec in specs:
            # Fetch core community fields. Property names come from the
            # registry; the chunk-only fields (keywords/excerpt) are
            # selected with NULL fallbacks so the result shape is uniform.
            if spec.kind == "chunk":
                summary_extras = (
                    "c.summary_keywords AS keywords, c.summary_excerpt AS excerpt"
                )
            else:
                summary_extras = "NULL AS keywords, NULL AS excerpt"

            cypher = (
                f"MATCH (c:`{spec.community_label}` "
                f"{{`{spec.id_property}`: $cid}}) "
                "WHERE c.graph_id IN $gids "
                f"RETURN c.`{spec.id_property}` AS community_id, "
                "c.summary AS summary, "
                f"c.`{spec.size_property}` AS size, "
                f"{summary_extras}"
            )
            result = await self._driver.execute_query(
                cypher, {"cid": community_id, "gids": gids}
            )
            if not result.records:
                continue

            rec = result.records[0]

            # Top-5 sample members. For entity, use ``name``; for chunk,
            # use a short text snippet so the agent has concrete evidence
            # without calling community_members for the full list.
            if spec.kind == "chunk":
                members_cypher = (
                    f"MATCH (m:`{spec.member_label}`)"
                    f"-[:`{spec.member_rel}`]->"
                    f"(c:`{spec.community_label}` "
                    f"{{`{spec.id_property}`: $cid}}) "
                    "WHERE m.graph_id IN $gids AND c.graph_id IN $gids "
                    "RETURN m.id AS member_id, "
                    "substring(coalesce(m.text, ''), 0, 80) AS member_preview "
                    "ORDER BY m.id LIMIT 5"
                )
            else:
                members_cypher = (
                    f"MATCH (m:`{spec.member_label}`)"
                    f"-[:`{spec.member_rel}`]->"
                    f"(c:`{spec.community_label}` "
                    f"{{`{spec.id_property}`: $cid}}) "
                    "WHERE m.graph_id IN $gids AND c.graph_id IN $gids "
                    "RETURN coalesce(m.id, elementId(m)) AS member_id, "
                    "coalesce(m.name, '') AS member_preview "
                    "ORDER BY m.name LIMIT 5"
                )
            mres = await self._driver.execute_query(
                members_cypher, {"cid": community_id, "gids": gids}
            )
            sample_members = [
                {"id": m["member_id"], "preview": m["member_preview"]}
                for m in mres.records
            ]

            # Parse keywords (chunk only).
            kw_raw = rec.get("keywords")
            keywords: list[str] | None = None
            if kw_raw:
                try:
                    import json as _json

                    parsed = _json.loads(kw_raw)
                    if isinstance(parsed, list):
                        keywords = parsed
                except (ValueError, TypeError):
                    keywords = None

            return [
                NodeResult(
                    id=rec["community_id"],
                    qualified_name=None,
                    label=(rec.get("summary") or "")[:160],
                    properties={
                        "kind": spec.kind,
                        "summary": rec.get("summary"),
                        "keywords": keywords,
                        "excerpt": rec.get("excerpt"),
                        "size": rec.get("size"),
                        "sample_members": sample_members,
                    },
                )
            ]

        # No kind matched
        return []

    # ── STORY-8: enriched retrieval tools (chat-engine unification) ──────────

    async def _vector_cypher_retrieve(
        self, gids: list[str], query: str, top_k: int
    ) -> list[NodeResult]:
        """Tenant-scoped vector retrieval + one-hop graph context.

        Cosine-scans the tenant's chunks (filter by ``graph_id`` first,
        score within — this avoids the cross-tenant top-K starvation of
        the global vector index), takes the ``top_k``, then pulls the
        connected entities and their one-hop relationships so the LLM
        gets graph-derived context, not just chunk text.

        Shared by :meth:`vector_cypher_search` and
        :meth:`hybrid_cypher_search`.
        """
        embedding = await self._embed(query)
        result = await self._driver.execute_query(
            """
            MATCH (node:Chunk)
            WHERE node.graph_id IN $gids AND node.embedding IS NOT NULL
            WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score
            ORDER BY score DESC
            LIMIT $top_k

            // graph-derived context for each retrieved chunk
            MATCH (entity)-[:FROM_CHUNK]->(node)
            WHERE entity.graph_id IN $gids
            OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(document)
                WHERE document.graph_id IN $gids
            OPTIONAL MATCH (entity)-[r]-(related_entity)
                WHERE related_entity.graph_id IN $gids

            RETURN node.text AS text,
                   document.path AS document_path,
                   collect(DISTINCT entity.name) AS entities,
                   collect(DISTINCT {
                       entity: related_entity.name,
                       relationship: type(r)
                   }) AS relationships,
                   score
            ORDER BY score DESC
            """,
            {"gids": gids, "embedding": embedding, "top_k": top_k},
        )
        return [_record_to_node_result(rec) for rec in result.records]

    async def vector_cypher_search(
        self,
        graph_id: str | list[str],
        query: str,
        top_k: int = 5,
    ) -> list[NodeResult]:
        """Vector similarity search + graph traversal (the 'enhanced' mode).

        For each chunk that scores highest by cosine similarity *within
        the tenant*, this also pulls connected entities and one-hop
        neighbour relationships so the LLM has graph-derived context,
        not just text. Default retriever for ``POST /chat`` after STORY-8.

        TASK-205: ``graph_id`` may be the effective graph-id set spanning
        LINKED_TO subgraphs; a one-element set is the single-graph case.

        Returns a list of NodeResult where ``label`` is the chunk text
        and ``properties`` carries ``score``, ``entities`` and
        ``relationships`` — the agent (or chat adapter) can quote these
        directly.
        """
        self._require("vector_cypher_search")
        gids = _as_graph_ids(graph_id)
        return await self._vector_cypher_retrieve(gids, query, top_k)

    async def hybrid_cypher_search(
        self,
        graph_id: str | list[str],
        query: str,
        top_k: int = 5,
    ) -> list[NodeResult]:
        """Vector search + graph traversal for the 'hybrid' chat modes.

        Shares the tenant-scoped vector-cypher retrieval with
        :meth:`vector_cypher_search`. The fulltext/BM25 half of the
        former hybrid retriever is intentionally not used: it ran through
        the global ``fulltext_chunks`` index with the same database-wide
        top-K that starved smaller tenants of results. Tenant-scoped
        fulltext needs its own design (``graph_id`` as an indexed
        fulltext field, or per-graph fulltext indexes) — until then
        ``hybrid`` returns correct vector-cypher results rather than
        starved ones.

        TASK-205: ``graph_id`` may be the effective graph-id set; a
        one-element set is the single-graph case.
        """
        self._require("hybrid_cypher_search")
        gids = _as_graph_ids(graph_id)
        return await self._vector_cypher_retrieve(gids, query, top_k)


def _record_to_node_result(record: Any) -> NodeResult:
    """Adapt a vector-cypher retrieval row to a ``NodeResult`` — the shape
    every other AgentToolkit method returns and the executor expects:
    ``label`` is the chunk text, ``properties`` carries ``score``,
    ``entities`` and ``relationships``.

    Chunk ids are not part of the retrieval projection, so a stable id is
    synthesized from the text hash — that keeps ``NodeResult.id`` non-empty
    for citation matching downstream.
    """
    import hashlib

    text = record.get("text") or ""
    synth_id = "chunk_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return NodeResult(
        id=synth_id,
        qualified_name=None,
        label=text[:200] if text else "(empty chunk)",
        properties={
            "text": text,
            "score": record.get("score"),
            "entities": [e for e in (record.get("entities") or []) if e],
            "relationships": [
                rel
                for rel in (record.get("relationships") or [])
                if rel and rel.get("entity")
            ],
            "document_path": record.get("document_path"),
        },
    )
