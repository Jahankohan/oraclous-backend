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
        """Return all nodes belonging to a Leiden community."""
        self._require("community_members")
        result = await self._driver.execute_query(
            """
            MATCH (n {graph_id: $gid})-[:BELONGS_TO]->(:Community {community_id: $cid})
            RETURN n
            LIMIT $max_results
            """,
            {"gid": graph_id, "cid": community_id, "max_results": max_results},
        )
        return [_node_to_result(dict(rec["n"])) for rec in result.records]

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
