"""
Oraclous MCP Server
===================
Thin adapter that exposes Oraclous knowledge graph operations via the Model
Context Protocol (MCP).  Graph CRUD, ingestion, and chat are delegated to the
Oraclous REST API over HTTP.  Low-level node inspection tools (search_nodes,
get_node, get_neighbors) issue direct Cypher queries against Neo4j because
those operations are not yet exposed as REST endpoints.

Environment variables
---------------------
ORACLOUS_API_KEY   – Bearer token / JWT used to authenticate against the
                     Oraclous API (required).
ORACLOUS_BASE_URL  – knowledge-graph-builder service root URL
                     (default: http://localhost:8003).
NEO4J_URI          – Bolt URI for direct Neo4j access used by node-inspection
                     tools (default: bolt://localhost:7687).
NEO4J_USERNAME     – (default: neo4j)
NEO4J_PASSWORD     – (default: password)
MCP_TRANSPORT      – 'stdio' (default) or 'sse'
MCP_HOST           – SSE bind host when transport=sse (default: 0.0.0.0)
MCP_PORT           – SSE port when transport=sse (default: 8004)

Usage
-----
# stdio (Claude Desktop / Cursor):
    python -m app.mcp.server

# SSE (Docker / remote):
    MCP_TRANSPORT=sse MCP_PORT=8004 python -m app.mcp.server
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Ensure the knowledge-graph-builder package root is on sys.path so that
# `from app.core.*` imports work when the server is launched as a subprocess.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ---------------------------------------------------------------------------
# Runtime config helpers
# ---------------------------------------------------------------------------


def _base_url() -> str:
    return os.environ.get("ORACLOUS_BASE_URL", "http://localhost:8003").rstrip("/")


def _api_key() -> str:
    key = os.environ.get("ORACLOUS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ORACLOUS_API_KEY is not set. "
            "Add it to your MCP client environment config."
        )
    return key


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {_api_key()}"}


# ---------------------------------------------------------------------------
# Shared async HTTP client (lazy, one per process)
# ---------------------------------------------------------------------------
_http_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=60.0)
    return _http_client


# ---------------------------------------------------------------------------
# Neo4j sync driver (lazy, used only by node-inspection tools)
# ---------------------------------------------------------------------------
_neo4j_driver = None
_neo4j_driver_owned = False  # True only when MCP created the driver itself


def _neo4j_sync_driver():
    """Return the shared sync Neo4j driver, initialising it on first call."""
    global _neo4j_driver, _neo4j_driver_owned
    if _neo4j_driver is not None:
        return _neo4j_driver

    # Prefer the app-level singleton when running inside the same process tree.
    try:
        from app.core.neo4j_client import neo4j_client as _app_client

        if _app_client.sync_driver is None:
            _app_client.connect_sync()
        _neo4j_driver = _app_client.sync_driver
        _neo4j_driver_owned = False  # borrowed — app manages lifecycle
        return _neo4j_driver
    except Exception:
        pass  # Fall back to standalone driver below

    # Standalone fallback — useful when the MCP server runs as its own process.
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    _neo4j_driver.verify_connectivity()
    _neo4j_driver_owned = True  # we created it — we must close it
    return _neo4j_driver


# ---------------------------------------------------------------------------
# Lifespan — clean up owned resources on server shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(server: FastMCP):
    """Async context manager that closes owned resources on MCP server shutdown."""
    yield
    global _neo4j_driver, _neo4j_driver_owned, _http_client
    if _neo4j_driver is not None and _neo4j_driver_owned:
        _neo4j_driver.close()
        _neo4j_driver = None
        _neo4j_driver_owned = False
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Oraclous",
    instructions=(
        "Use these tools to create and query knowledge graphs on Oraclous. "
        "Start by calling list_graphs to see existing graphs, or create_graph "
        "to make a new one.  Use ingest_text to populate the graph with "
        "documents or notes, then call chat to ask natural language questions "
        "grounded in the extracted entities and relationships."
    ),
    lifespan=_lifespan,
)

# ===========================================================================
# 1 – Graph Management
# ===========================================================================


@mcp.tool()
async def create_graph(name: str, description: str = "") -> dict:
    """
    Create a new knowledge graph.

    Args:
        name: Human-readable name for the graph.
        description: Optional description of what this graph will contain.

    Returns:
        Graph metadata including graph_id, name, status, node_count,
        and relationship_count.
    """
    resp = await _client().post(
        f"{_base_url()}/api/v1/graphs",
        headers=_auth_headers(),
        json={"name": name, "description": description},
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "graph_id": str(data["id"]),
        "name": data["name"],
        "description": data.get("description", ""),
        "status": data.get("status", "active"),
        "node_count": data.get("node_count", 0),
        "relationship_count": data.get("relationship_count", 0),
    }


@mcp.tool()
async def list_graphs() -> list:
    """
    List all knowledge graphs owned by the authenticated user.

    Returns:
        List of graph objects, each with graph_id, name, description,
        node_count, and relationship_count.
    """
    resp = await _client().get(
        f"{_base_url()}/api/v1/graphs",
        headers=_auth_headers(),
    )
    resp.raise_for_status()
    return [
        {
            "graph_id": str(g["id"]),
            "name": g["name"],
            "description": g.get("description", ""),
            "status": g.get("status", "active"),
            "node_count": g.get("node_count", 0),
            "relationship_count": g.get("relationship_count", 0),
        }
        for g in resp.json()
    ]


@mcp.tool()
async def delete_graph(graph_id: str) -> dict:
    """
    Permanently delete a knowledge graph and all its nodes and relationships.

    Ownership is verified before deletion — you can only delete graphs that
    belong to your account.

    Args:
        graph_id: UUID of the graph to delete.

    Returns:
        Confirmation dict with graph_id and deleted (bool).
    """
    # Verify ownership via the GET endpoint first.
    check = await _client().get(
        f"{_base_url()}/api/v1/graphs/{graph_id}",
        headers=_auth_headers(),
    )
    if check.status_code == 404:
        return {"graph_id": graph_id, "deleted": False, "error": "Graph not found."}
    if check.status_code == 403:
        return {"graph_id": graph_id, "deleted": False, "error": "Access denied."}
    check.raise_for_status()

    user_id = check.json().get("user_id")

    # Use GraphNodeService directly (no REST DELETE endpoint yet).
    try:
        from app.core.neo4j_client import neo4j_client as _app_client
        from app.services.graph_node_service import GraphNodeService

        if _app_client.sync_driver is None:
            _app_client.connect_sync()
        svc = GraphNodeService(_app_client.sync_driver)
        deleted = svc.delete_graph(graph_id=graph_id, user_id=str(user_id))
    except Exception as exc:
        return {"graph_id": graph_id, "deleted": False, "error": str(exc)}

    return {"graph_id": graph_id, "deleted": deleted}


@mcp.tool()
async def get_graph_stats(graph_id: str) -> dict:
    """
    Return statistics for a knowledge graph.

    Args:
        graph_id: UUID of the graph.

    Returns:
        Stats dict with name, status, node_count, and relationship_count.
    """
    resp = await _client().get(
        f"{_base_url()}/api/v1/graphs/{graph_id}",
        headers=_auth_headers(),
    )
    if resp.status_code == 404:
        return {"error": f"Graph {graph_id!r} not found."}
    if resp.status_code == 403:
        return {"error": "Access denied."}
    resp.raise_for_status()
    data = resp.json()
    return {
        "graph_id": str(data["id"]),
        "name": data["name"],
        "description": data.get("description", ""),
        "status": data.get("status", "active"),
        "node_count": data.get("node_count", 0),
        "relationship_count": data.get("relationship_count", 0),
    }


# ===========================================================================
# 2 – Data Ingestion
# ===========================================================================


@mcp.tool()
async def ingest_text(
    graph_id: str,
    text: str,
    source_label: str = "",
    context: str = "",
) -> dict:
    """
    Ingest text content into a knowledge graph.

    Entities and relationships are automatically extracted by the LLM
    pipeline and stored in Neo4j.  The job runs asynchronously — call
    get_graph_stats after a few seconds to check updated node/edge counts.

    Args:
        graph_id: UUID of the target graph.
        text: Raw text to ingest (documents, notes, articles, meeting
              transcripts, etc.).
        source_label: Optional human-readable label for this content source.
        context: Optional domain hint to guide LLM extraction, e.g.
                 'HR employee records' or 'pharmaceutical research notes'.

    Returns:
        Job info with job_id, status, and graph_id.
    """
    payload: dict = {"content": text, "source_type": "text"}
    overrides: dict = {}
    if source_label:
        overrides["source_label"] = source_label
    if context:
        overrides["additional_focus"] = context
    if overrides:
        payload["overrides"] = overrides

    resp = await _client().post(
        f"{_base_url()}/api/v1/graphs/{graph_id}/ingest",
        headers=_auth_headers(),
        json=payload,
    )
    if resp.status_code == 404:
        return {"error": f"Graph {graph_id!r} not found."}
    if resp.status_code == 403:
        return {"error": "Access denied."}
    resp.raise_for_status()
    data = resp.json()
    return {
        "job_id": str(data["id"]),
        "graph_id": str(data["graph_id"]),
        "status": data.get("status", "pending"),
        "message": "Ingestion started. Call get_graph_stats in a few seconds to track progress.",
    }


@mcp.tool()
async def ingest_file(graph_id: str, file_path: str, context: str = "") -> dict:
    """
    Read a local file and ingest its text contents into a knowledge graph.

    The file is read as plain text.  For best results with structured
    documents, convert them to plain text before ingesting.

    Args:
        graph_id: UUID of the target graph.
        file_path: Absolute or relative path to the file on the local
                   filesystem.
        context: Optional domain hint for LLM extraction.

    Returns:
        Job info with job_id, status, graph_id, and the resolved file path.
    """
    fp = Path(file_path).expanduser().resolve()
    if not fp.exists():
        return {"error": f"File not found: {file_path!r}"}

    try:
        content = fp.read_text(errors="replace")
    except Exception as exc:
        return {"error": f"Cannot read file {file_path!r}: {exc}"}

    payload: dict = {
        "content": content,
        "source_type": fp.suffix.lstrip(".") or "text",
    }
    if context:
        payload["overrides"] = {"additional_focus": context}

    resp = await _client().post(
        f"{_base_url()}/api/v1/graphs/{graph_id}/ingest",
        headers=_auth_headers(),
        json=payload,
    )
    if resp.status_code == 404:
        return {"error": f"Graph {graph_id!r} not found."}
    if resp.status_code == 403:
        return {"error": "Access denied."}
    resp.raise_for_status()
    data = resp.json()
    return {
        "job_id": str(data["id"]),
        "graph_id": str(data["graph_id"]),
        "status": data.get("status", "pending"),
        "file": str(fp),
        "message": "Ingestion started. Call get_graph_stats in a few seconds to track progress.",
    }


# ===========================================================================
# 3 – Query & Chat
# ===========================================================================


@mcp.tool()
async def chat(
    graph_id: str,
    question: str,
    mode: str = "enhanced",
) -> dict:
    """
    Ask a natural language question answered using the knowledge graph.

    Responses are strictly grounded in retrieved graph data.  When the graph
    does not contain sufficient context, the answer will say so rather than
    speculating.

    Args:
        graph_id: UUID of the graph to query.
        question: Natural language question.
        mode: Retrieval strategy — one of:
              'enhanced'    (default) vector search + graph traversal,
              'simple'      pure vector similarity search (fastest),
              'hybrid'      vector + full-text search,
              'hybrid_plus' hybrid with graph traversal,
              'natural'     natural language translated to Cypher.

    Returns:
        Dict with answer (str), sources (list), is_grounded (bool),
        and retriever_used (str).
    """
    valid_modes = {"simple", "enhanced", "hybrid", "hybrid_plus", "natural"}
    if mode not in valid_modes:
        return {
            "error": (
                f"Invalid mode {mode!r}. "
                f"Choose from: {', '.join(sorted(valid_modes))}"
            )
        }

    resp = await _client().post(
        f"{_base_url()}/api/v1/chat",
        headers=_auth_headers(),
        json={
            "query": question,
            "graph_id": graph_id,
            "mode": mode,
            "include_sources": True,
        },
    )
    if resp.status_code == 403:
        return {"error": "Graph not found or access denied."}
    resp.raise_for_status()
    data = resp.json()
    return {
        "answer": data.get("answer", ""),
        "is_grounded": data.get("is_grounded", False),
        "sources": data.get("sources", []),
        "retriever_used": data.get("retriever_used", mode),
    }


# ===========================================================================
# 4 – Node Inspection  (direct Neo4j queries — no REST endpoint yet)
# ===========================================================================


async def _assert_graph_access(graph_id: str) -> bool:
    """Return True if the authenticated user can access graph_id, else False."""
    resp = await _client().get(
        f"{_base_url()}/api/v1/graphs/{graph_id}",
        headers=_auth_headers(),
    )
    return resp.status_code == 200


@mcp.tool()
async def search_nodes(
    graph_id: str,
    query: str,
    entity_type: str = "",
    limit: int = 10,
) -> list:
    """
    Search for entities in a knowledge graph by name or description.

    Uses case-insensitive substring matching.  For semantic search over
    entity embeddings, use the chat tool instead.

    Args:
        graph_id: UUID of the graph to search.
        query: Search string matched against entity name and description.
        entity_type: Optional filter — return only entities of this type
                     (e.g. 'Person', 'Company', 'Drug').
        limit: Maximum results to return (default 10, max 50).

    Returns:
        List of entity dicts with entity_id, name, type, and description.
    """
    if not await _assert_graph_access(graph_id):
        return [{"error": "Graph not found or access denied."}]

    limit = min(max(1, limit), 50)
    params: dict = {"graph_id": graph_id, "query": query, "limit": limit}

    cypher = (
        "MATCH (e:__Entity__ {graph_id: $graph_id}) "
        "WHERE toLower(e.name) CONTAINS toLower($query) "
        "   OR (e.description IS NOT NULL "
        "       AND toLower(e.description) CONTAINS toLower($query)) "
    )
    if entity_type:
        cypher += "AND e.type = $entity_type "
        params["entity_type"] = entity_type

    cypher += (
        "RETURN e.entity_id AS entity_id, e.name AS name, "
        "       e.type AS type, e.description AS description "
        "ORDER BY e.name "
        "LIMIT $limit"
    )

    driver = _neo4j_sync_driver()
    results = []
    with driver.session() as session:
        for record in session.run(cypher, params):
            results.append(
                {
                    "entity_id": record["entity_id"],
                    "name": record["name"],
                    "type": record.get("type") or "",
                    "description": record.get("description") or "",
                }
            )
    return results


@mcp.tool()
async def get_node(graph_id: str, entity_name: str) -> dict:
    """
    Retrieve a specific entity node by name from a knowledge graph.

    Performs a case-insensitive exact match on the entity name.

    Args:
        graph_id: UUID of the graph.
        entity_name: Name of the entity to retrieve.

    Returns:
        Dict of entity properties, or an error message if not found.
    """
    if not await _assert_graph_access(graph_id):
        return {"error": "Graph not found or access denied."}

    driver = _neo4j_sync_driver()
    cypher = (
        "MATCH (e:__Entity__ {graph_id: $graph_id}) "
        "WHERE toLower(e.name) = toLower($name) "
        "RETURN e LIMIT 1"
    )
    with driver.session() as session:
        record = session.run(
            cypher, {"graph_id": graph_id, "name": entity_name}
        ).single()

    if not record:
        return {"error": (f"Entity {entity_name!r} not found in graph {graph_id!r}.")}

    node = dict(record["e"])
    node.pop("embedding", None)  # strip large vector field
    return node


@mcp.tool()
async def get_neighbors(
    graph_id: str,
    entity_name: str,
    hops: int = 1,
) -> dict:
    """
    Get an entity and all its directly connected neighbours in the graph.

    Args:
        graph_id: UUID of the graph.
        entity_name: Name of the anchor entity.
        hops: Traversal depth — 1 (immediate neighbours) or 2 (two-hop).
              Values outside this range are clamped.

    Returns:
        Dict with entity (str), graph_id, hops, and neighbors (list of
        dicts describing each connected node and its relationship type).
    """
    if not await _assert_graph_access(graph_id):
        return {"error": "Graph not found or access denied."}

    hops = min(max(1, hops), 2)
    driver = _neo4j_sync_driver()
    params: dict = {"graph_id": graph_id, "name": entity_name}

    if hops == 1:
        cypher = """
        MATCH (anchor:__Entity__ {graph_id: $graph_id})
        WHERE toLower(anchor.name) = toLower($name)
        OPTIONAL MATCH (anchor)-[r]->(nb:__Entity__ {graph_id: $graph_id})
        RETURN anchor.name AS anchor,
               type(r)     AS rel_type,
               nb.name     AS neighbor,
               nb.type     AS neighbor_type
        LIMIT 50
        """
    else:
        cypher = """
        MATCH (anchor:__Entity__ {graph_id: $graph_id})
        WHERE toLower(anchor.name) = toLower($name)
        OPTIONAL MATCH (anchor)-[r1]->(h1:__Entity__ {graph_id: $graph_id})
        OPTIONAL MATCH (h1)-[r2]->(h2:__Entity__ {graph_id: $graph_id})
        RETURN anchor.name AS anchor,
               type(r1)    AS rel1,
               h1.name     AS hop1_name,
               h1.type     AS hop1_type,
               type(r2)    AS rel2,
               h2.name     AS hop2_name,
               h2.type     AS hop2_type
        LIMIT 50
        """

    anchor_name: str | None = None
    neighbors: list = []
    with driver.session() as session:
        for record in session.run(cypher, params):
            if anchor_name is None:
                anchor_name = record["anchor"]
            if hops == 1:
                if record["neighbor"]:
                    neighbors.append(
                        {
                            "relationship": record["rel_type"],
                            "name": record["neighbor"],
                            "type": record.get("neighbor_type") or "",
                            "hop": 1,
                        }
                    )
            else:
                if record["hop1_name"]:
                    neighbors.append(
                        {
                            "relationship": record["rel1"],
                            "name": record["hop1_name"],
                            "type": record.get("hop1_type") or "",
                            "hop": 1,
                        }
                    )
                if record["hop2_name"]:
                    neighbors.append(
                        {
                            "relationship": record["rel2"],
                            "name": record["hop2_name"],
                            "type": record.get("hop2_type") or "",
                            "hop": 2,
                        }
                    )

    if anchor_name is None:
        return {"error": (f"Entity {entity_name!r} not found in graph {graph_id!r}.")}

    return {
        "entity": anchor_name,
        "graph_id": graph_id,
        "hops": hops,
        "neighbors": neighbors,
    }


# ===========================================================================
# 5 – MCP Resources
# ===========================================================================


@mcp.resource("graphs://")
async def resource_graphs() -> str:
    """All knowledge graphs accessible to the authenticated user."""
    try:
        graphs = await list_graphs()
    except Exception as exc:
        return f"Error fetching graphs: {exc}"

    if not graphs:
        return "No knowledge graphs found for this account."

    lines = ["# Available Knowledge Graphs\n"]
    for g in graphs:
        lines.append(
            f"- **{g['name']}** (`{g['graph_id']}`)\n"
            f"  Status: {g['status']} | "
            f"Nodes: {g['node_count']} | "
            f"Relationships: {g['relationship_count']}"
        )
        if g.get("description"):
            lines.append(f"  {g['description']}")
    return "\n".join(lines)


@mcp.resource("graph://{graph_id}/stats")
async def resource_graph_stats(graph_id: str) -> str:
    """Statistics summary for a specific knowledge graph."""
    stats = await get_graph_stats(graph_id)
    if "error" in stats:
        return f"Error: {stats['error']}"
    return (
        f"# {stats['name']}\n\n"
        f"- **ID**: {stats['graph_id']}\n"
        f"- **Status**: {stats['status']}\n"
        f"- **Nodes**: {stats['node_count']}\n"
        f"- **Relationships**: {stats['relationship_count']}\n"
        f"- **Description**: {stats.get('description') or 'N/A'}"
    )


@mcp.resource("graph://{graph_id}/nodes")
async def resource_graph_nodes(graph_id: str) -> str:
    """Sample of up to 50 entity nodes in a knowledge graph."""
    if not await _assert_graph_access(graph_id):
        return "Graph not found or access denied."

    try:
        driver = _neo4j_sync_driver()
    except Exception as exc:
        return f"Neo4j unavailable: {exc}"

    cypher = (
        "MATCH (e:__Entity__ {graph_id: $graph_id}) "
        "RETURN e.name AS name, e.type AS type, e.description AS description "
        "ORDER BY e.name LIMIT 50"
    )
    lines = [f"# Entities in graph `{graph_id}`\n"]
    with driver.session() as session:
        for record in session.run(cypher, {"graph_id": graph_id}):
            entry = f"- **{record['name']}** ({record.get('type') or '?'})"
            if record.get("description"):
                entry += f" — {record['description']}"
            lines.append(entry)

    if len(lines) == 1:
        lines.append("No entities found yet. Use ingest_text to populate this graph.")
    return "\n".join(lines)


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if transport == "sse":
        host = os.environ.get("MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_PORT", "8004"))
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
