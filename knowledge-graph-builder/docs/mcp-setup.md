# Oraclous MCP Server — Setup Guide

Connect Claude Desktop, Cursor, Continue, or any MCP-compatible AI client to your Oraclous knowledge graphs.

---

## What Is the MCP Server?

The Oraclous MCP server exposes your knowledge graph operations as tools that AI assistants can call directly.  Once connected, Claude (or any MCP client) can:

- Create and manage knowledge graphs
- Ingest text, files, and web pages into a graph
- Ask natural language questions answered from your graph
- Browse individual entities and their relationships

The server communicates over either **stdio** (local subprocess, used by Claude Desktop and Cursor) or **SSE** (HTTP server, used by Docker-based deployments).

---

## Prerequisites

- Python 3.11+ with the `knowledge-graph-builder` dependencies installed (`pip install -r requirements.txt`)
- A running Oraclous API (`knowledge-graph-builder` service on port 8003, or remote)
- Your Oraclous API key (a JWT from the auth service, or a static key if configured)

---

## Quick Start (stdio — local)

### 1. Set environment variables

```bash
export ORACLOUS_API_KEY="<your-api-key>"
export ORACLOUS_BASE_URL="http://localhost:8003"   # default; change for remote
export NEO4J_URI="bolt://localhost:7687"            # optional — only needed for node-inspection tools
export NEO4J_USERNAME="neo4j"                       # optional
export NEO4J_PASSWORD="password"                    # optional
```

### 2. Verify the server starts

```bash
cd /path/to/oraclous-data-studio/knowledge-graph-builder
python -m app.mcp.server
# You should see FastMCP startup output with all tools listed.
# Press Ctrl+C to stop.
```

---

## Claude Desktop

Add the following block to your `claude_desktop_config.json` (usually at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "oraclous": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "/path/to/oraclous-data-studio/knowledge-graph-builder",
      "env": {
        "ORACLOUS_API_KEY": "<your-api-key>",
        "ORACLOUS_BASE_URL": "http://localhost:8003",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password"
      }
    }
  }
}
```

Restart Claude Desktop.  You should see **Oraclous** appear in the tool list.

---

## Cursor

Open Cursor settings → **MCP** → **Add MCP Server** and fill in:

| Field | Value |
|---|---|
| Name | `oraclous` |
| Type | `command` |
| Command | `python -m app.mcp.server` |
| Working directory | `/path/to/oraclous-data-studio/knowledge-graph-builder` |
| Environment variables | `ORACLOUS_API_KEY`, `ORACLOUS_BASE_URL`, etc. |

Alternatively, add it directly to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "oraclous": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "/path/to/oraclous-data-studio/knowledge-graph-builder",
      "env": {
        "ORACLOUS_API_KEY": "<your-api-key>",
        "ORACLOUS_BASE_URL": "http://localhost:8003"
      }
    }
  }
}
```

---

## Docker (SSE transport)

The `knowledge-graph-mcp` service in `docker-compose.yml` runs the server in SSE mode on port **8004**.

```bash
docker compose up knowledge-graph-mcp
```

To connect a client to the SSE endpoint:

```json
{
  "mcpServers": {
    "oraclous": {
      "type": "sse",
      "url": "http://localhost:8004/sse",
      "env": {
        "ORACLOUS_API_KEY": "<your-api-key>"
      }
    }
  }
}
```

> **Note:** SSE transport requires MCP client support for remote servers.  Claude Desktop currently supports stdio only; use SSE with Continue, Cline, or custom integrations.

---

## Available Tools

| Tool | Description |
|---|---|
| `create_graph` | Create a new knowledge graph |
| `list_graphs` | List all your graphs |
| `delete_graph` | Delete a graph permanently |
| `get_graph_stats` | Node/edge counts and graph metadata |
| `ingest_text` | Ingest raw text; entities extracted automatically |
| `ingest_file` | Read and ingest a local file |
| `chat` | Natural language Q&A grounded in the graph |
| `search_nodes` | Search entities by name/description substring |
| `get_node` | Retrieve a single entity by name |
| `get_neighbors` | Get an entity's connected neighbours (1–2 hops) |

## Available Resources

| Resource URI | Description |
|---|---|
| `graphs://` | List of all your knowledge graphs |
| `graph://{graph_id}/stats` | Statistics for a specific graph |
| `graph://{graph_id}/nodes` | Sample of up to 50 entities in a graph |

---

## Example Conversation

Once connected, you can use Claude naturally:

> **You:** Create a knowledge graph called "Research Notes" and ingest this text: "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity..."

> **Claude:** *(calls `create_graph`, then `ingest_text`)* Done! Created the "Research Notes" graph. The ingestion job has started — entities like *Marie Curie*, *radioactivity*, and related concepts are being extracted.

> **You:** Who was Marie Curie married to?

> **Claude:** *(calls `chat`)* According to your knowledge graph, Marie Curie was married to Pierre Curie, a fellow physicist with whom she collaborated on radioactivity research.

---

## Troubleshooting

### "ORACLOUS_API_KEY is not set"
Set the `ORACLOUS_API_KEY` environment variable in your MCP client config.

### "Graph not found or access denied"
Verify your API key has access to the graph.  Call `list_graphs` to see accessible graphs.

### Neo4j connection errors for `search_nodes`, `get_node`, `get_neighbors`
These tools use direct Neo4j access.  Set `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`.  If the server cannot reach Neo4j, the REST-based tools (`chat`, `ingest_text`, etc.) still work.

### Server doesn't appear in Claude Desktop
1. Confirm the `cwd` path is correct and the Python environment has `mcp` installed.
2. Check the Claude Desktop developer console for error output.
3. Run `python -m app.mcp.server` manually to verify there are no import errors.

---

## Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| `ORACLOUS_API_KEY` | *(required)* | Bearer token for the Oraclous API |
| `ORACLOUS_BASE_URL` | `http://localhost:8003` | Oraclous API base URL |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `MCP_TRANSPORT` | `stdio` | Transport mode: `stdio` or `sse` |
| `MCP_HOST` | `0.0.0.0` | SSE bind host (SSE mode only) |
| `MCP_PORT` | `8004` | SSE port (SSE mode only) |
