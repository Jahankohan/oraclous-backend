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
        self, graph_id: str, query: str, max_results: int = 10
    ) -> list[NodeResult]:
        """Semantic similarity search over the graph's embedding space."""
        self._require("graph_search")
        embedding = await self._embed(query)
        result = await self._driver.execute_query(
            """
            CALL db.index.vector.queryNodes('text_embeddings_primary', $max_results, $embedding)
            YIELD node, score
            WHERE node.graph_id = $gid
            RETURN node, score
            ORDER BY score DESC
            """,
            {"gid": graph_id, "max_results": max_results, "embedding": embedding},
        )
        return [_node_to_result(dict(rec["node"])) for rec in result.records]

    async def community_members(
        self, graph_id: str, community_id: str, max_results: int = 50
    ) -> list[NodeResult]:
        """Return all member nodes of a community across any registered kind.

        Each registered kind in ``COMMUNITY_KINDS`` is probed in order until
        a match for ``community_id`` is found. This avoids forcing the
        calling LLM to know whether ``community_id`` points at an entity
        community or a chunk community.
        """
        self._require("community_members")
        # Imported lazily so the toolkit can still be used in tests that
        # don't have the full app package wired up.
        from app.schemas.community_kinds import all_kinds

        for spec in all_kinds():
            # Labels, relationship types, and property names come from the
            # compile-time registry — never from user input — so f-string
            # interpolation here is safe. Identifiers wrapped in backticks
            # for defense-in-depth.
            query = (
                f"MATCH (n:`{spec.member_label}` {{graph_id: $gid}})"
                f"-[:`{spec.member_rel}`]->"
                f"(c:`{spec.community_label}` "
                f"{{`{spec.id_property}`: $cid, graph_id: $gid}}) "
                "RETURN n "
                "LIMIT $max_results"
            )
            result = await self._driver.execute_query(
                query,
                {"gid": graph_id, "cid": community_id, "max_results": max_results},
            )
            if result.records:
                return [_node_to_result(dict(rec["n"])) for rec in result.records]

        return []

    async def neighbors(
        self,
        graph_id: str,
        node_id: str,
        edge_type: str | None = None,
        depth: int = 1,
    ) -> list[NodeResult]:
        """BFS from node_id up to `depth` hops, optionally filtered by relationship type."""
        self._require("neighbors")
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
        self, graph_id: str, node_label: str, top_n: int = 10
    ) -> list[NodeResult]:
        """Return top_n most-connected nodes of the given label (Cypher COUNT, not GDS)."""
        self._require("degree_centrality")
        label = _safe_label(node_label)
        # label cannot be a Cypher parameter; validated above
        query = f"""
        MATCH (n:{label} {{graph_id: $gid}})-[r]-()
        RETURN n, count(r) AS degree
        ORDER BY degree DESC
        LIMIT $top_n
        """
        result = await self._driver.execute_query(
            query, {"gid": graph_id, "top_n": top_n}
        )
        return [_node_to_result(dict(rec["n"])) for rec in result.records]

    async def shortest_path(
        self, graph_id: str, from_qname: str, to_qname: str
    ) -> PathResult | None:
        """Return shortest undirected path between two nodes by qualified_name."""
        self._require("shortest_path")
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
        self, graph_id: str, source_qname: str, depth: int = 10
    ) -> list[NodeResult]:
        """Follow FLOWS_TO edges from source_qname (valid for code knowledge graphs)."""
        self._require("taint_trace")
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
        graph_id: str,
        cypher: str,
        max_results: int = 25,
    ) -> list[NodeResult]:
        """Execute a read-only Cypher query scoped to this graph.

        The agent's LLM generates Cypher (text2cypher pattern). Safety rules:
          - Block any write operation (CREATE, DELETE, SET, REMOVE, MERGE, DROP).
          - Block APOC procedures that can mutate.
          - Require the query to constrain by ``graph_id`` (tenant isolation).
          - Cap result size at ``max_results`` via injected LIMIT clause if absent.

        Returns the first node-valued column from each record.
        """
        self._require("cypher_query")
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
        graph_id: str,
        node_label: str,
        at_time: int,
        max_results: int = 50,
    ) -> list[NodeResult]:
        """Return nodes valid at a given point in time (bitemporal valid_from/valid_to)."""
        self._require("temporal_slice")
        label = _safe_label(node_label)
        # label cannot be a Cypher parameter; validated above
        query = f"""
        MATCH (n:{label} {{graph_id: $gid}})
        WHERE n.valid_from <= $at_time
          AND (n.valid_to IS NULL OR n.valid_to >= $at_time)
        RETURN n
        LIMIT $max_results
        """
        result = await self._driver.execute_query(
            query, {"gid": graph_id, "at_time": at_time, "max_results": max_results}
        )
        return [_node_to_result(dict(rec["n"])) for rec in result.records]

    # ── STORY-4d: community discovery + description tools ────────────────────

    async def find_communities(
        self,
        graph_id: str,
        query: str,
        kind: str | None = None,
        top_k: int = 5,
    ) -> list[NodeResult]:
        """Find communities related to a query via vector search over summaries.

        ``kind=None`` searches every registered kind that has a populated
        vector index, then merges results by score. ``kind="chunk"`` or
        ``kind="entity"`` restricts to one kind.

        Each result's ``properties`` carries ``kind``, ``score``,
        ``summary``, ``keywords`` (chunk only), ``excerpt`` (chunk only),
        ``size``. The ``label`` field surfaces the summary preview so
        agent prompts can read the result in a glance.
        """
        self._require("find_communities")
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
                f"WHERE node.graph_id = $gid AND node.summary IS NOT NULL "
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
                        "gid": graph_id,
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
        graph_id: str,
        community_id: str,
        kind: str | None = None,
    ) -> list[NodeResult]:
        """Return metadata for one community: summary, keywords, excerpt,
        size, plus up to 5 sample member names.

        Auto-detects kind by probing the registry (entity first → one
        round trip on the common case) when ``kind`` is None. The agent
        can also pass ``kind`` explicitly to skip the probe.

        Returns a single-element ``list[NodeResult]`` for shape
        consistency with the other agent tools. The result's
        ``properties`` carry every relevant field; the ``label`` is a
        summary preview.
        """
        self._require("describe_community")
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
                f"{{`{spec.id_property}`: $cid, graph_id: $gid}}) "
                f"RETURN c.`{spec.id_property}` AS community_id, "
                "c.summary AS summary, "
                f"c.`{spec.size_property}` AS size, "
                f"{summary_extras}"
            )
            result = await self._driver.execute_query(
                cypher, {"cid": community_id, "gid": graph_id}
            )
            if not result.records:
                continue

            rec = result.records[0]

            # Top-5 sample members. For entity, use ``name``; for chunk,
            # use a short text snippet so the agent has concrete evidence
            # without calling community_members for the full list.
            if spec.kind == "chunk":
                members_cypher = (
                    f"MATCH (m:`{spec.member_label}` {{graph_id: $gid}})"
                    f"-[:`{spec.member_rel}`]->"
                    f"(c:`{spec.community_label}` "
                    f"{{`{spec.id_property}`: $cid, graph_id: $gid}}) "
                    "RETURN m.id AS member_id, "
                    "substring(coalesce(m.text, ''), 0, 80) AS member_preview "
                    "ORDER BY m.id LIMIT 5"
                )
            else:
                members_cypher = (
                    f"MATCH (m:`{spec.member_label}` {{graph_id: $gid}})"
                    f"-[:`{spec.member_rel}`]->"
                    f"(c:`{spec.community_label}` "
                    f"{{`{spec.id_property}`: $cid, graph_id: $gid}}) "
                    "RETURN coalesce(m.id, elementId(m)) AS member_id, "
                    "coalesce(m.name, '') AS member_preview "
                    "ORDER BY m.name LIMIT 5"
                )
            mres = await self._driver.execute_query(
                members_cypher, {"cid": community_id, "gid": graph_id}
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

    async def vector_cypher_search(
        self,
        graph_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[NodeResult]:
        """Vector similarity search + graph traversal (the 'enhanced' mode).

        Wraps neo4j_graphrag's VectorCypherRetriever. For each chunk that
        scores above the vector threshold, this also pulls connected
        entities and one-hop neighbour relationships so the LLM has
        graph-derived context, not just text. This is the default
        retriever for ``POST /chat`` after STORY-8.

        Returns a list of NodeResult where each result's ``label`` is the
        chunk text and ``properties`` carries ``score``, ``entities``,
        and ``relationships``. The agent (or chat adapter) can quote
        these directly.
        """
        self._require("vector_cypher_search")
        # Imported lazily so the toolkit can be used in tests without
        # constructing the full retriever stack.
        from app.schemas.retriever_schemas import (
            DefaultRetrieverConfigs,
            RetrieverConfig,
            RetrieverType,
        )
        from app.services.retriever_factory import retriever_factory

        cfg = DefaultRetrieverConfigs.get_vector_cypher_config(graph_id)
        cfg.top_k = top_k
        retriever = await retriever_factory.create_retriever(
            RetrieverConfig(type=RetrieverType.VECTOR_CYPHER, config=cfg),
            graph_id,
        )
        # neo4j_graphrag retrievers are sync — call via to_thread to
        # avoid blocking the executor's event loop. Explicit
        # ``query_text=`` keyword — the first positional arg on
        # ``search`` is the embedding vector, not the query string.
        import asyncio as _asyncio

        def _do_search():
            return retriever.search(
                query_text=query,
                top_k=top_k,
                query_params={"graph_id": graph_id},
            )

        result = await _asyncio.to_thread(_do_search)
        return _retriever_items_to_node_results(result)

    async def hybrid_cypher_search(
        self,
        graph_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[NodeResult]:
        """Hybrid (vector + fulltext) search + graph traversal.

        Wraps neo4j_graphrag's HybridCypherRetriever. Like
        ``vector_cypher_search`` but the initial retrieval is a hybrid
        of vector similarity AND fulltext BM25 matching, so exact-term
        queries (proper nouns, identifiers) get caught alongside
        semantic matches. The graph-traversal half is identical to the
        vector-cypher version.

        Requires the ``fulltext_chunks`` index to exist. If it doesn't,
        the underlying retriever will error — callers should fall back
        to ``vector_cypher_search`` on failure.
        """
        self._require("hybrid_cypher_search")
        from app.schemas.retriever_schemas import (
            DefaultRetrieverConfigs,
            RetrieverConfig,
            RetrieverType,
        )
        from app.services.retriever_factory import retriever_factory

        cfg = DefaultRetrieverConfigs.get_hybrid_cypher_config(graph_id)
        cfg.top_k = top_k
        retriever = await retriever_factory.create_retriever(
            RetrieverConfig(type=RetrieverType.HYBRID_CYPHER, config=cfg),
            graph_id,
        )
        import asyncio as _asyncio

        def _do_search():
            return retriever.search(
                query_text=query,
                top_k=top_k,
                query_params={"graph_id": graph_id},
            )

        result = await _asyncio.to_thread(_do_search)
        return _retriever_items_to_node_results(result)


def _retriever_items_to_node_results(retriever_result: Any) -> list[NodeResult]:
    """Adapt a neo4j_graphrag RetrieverResult to ``list[NodeResult]`` —
    the shape every other AgentToolkit method returns and the executor
    expects.

    RetrieverResult.items each have ``content`` (the chunk text, possibly
    formatted with embedded entity/relationship metadata) and
    ``metadata`` (a dict with at minimum a ``score``). The chunk's
    Neo4j id isn't carried through neo4j_graphrag's surface, so we
    synthesize a stable id from the content hash. That keeps the
    NodeResult.id non-empty for citation matching downstream.
    """
    import hashlib

    items = getattr(retriever_result, "items", None) or []
    out: list[NodeResult] = []
    for item in items:
        content = getattr(item, "content", "") or ""
        metadata = dict(getattr(item, "metadata", {}) or {})
        # Synthesize a stable id so downstream citation matching has
        # something to anchor on. Real chunk ids aren't exposed by the
        # neo4j_graphrag retriever surface.
        synth_id = "chunk_" + hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
        out.append(
            NodeResult(
                id=synth_id,
                qualified_name=None,
                label=content[:200] if content else "(empty chunk)",
                properties={
                    "text": content,
                    **metadata,
                },
            )
        )
    return out
