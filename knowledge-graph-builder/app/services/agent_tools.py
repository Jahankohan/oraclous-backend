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
