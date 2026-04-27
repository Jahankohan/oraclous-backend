# Quickstart: Zero to First Knowledge Graph in 10 Minutes

This guide takes you from a fresh checkout to querying your first knowledge graph with natural language.

**What you'll build:** A running Knowledge Graph Builder service, with a graph created, a document ingested, and a chat query answered.

---

## Prerequisites

Before you begin, make sure you have:

- **Docker** and **Docker Compose** installed (Docker 20.10+ recommended)
- An **OpenAI API key** (for entity extraction and chat) — or an Anthropic API key
- **curl** or a REST client for testing the API
- Git to clone the repository

---

## Step 1 — Clone and Configure

```bash
git clone https://github.com/oraclous/oraclous-data-studio.git
cd oraclous-data-studio
```

Copy the environment template for the knowledge graph builder:

```bash
cp knowledge-graph-builder/.env.example knowledge-graph-builder/.env
```

Open `knowledge-graph-builder/.env` and set your LLM key:

```bash
# Required: at least one LLM provider
OPENAI_API_KEY=sk-...your-key-here...

# Optional: leave other values as defaults for local development
```

The default configuration uses:
- Neo4j at `bolt://neo4j:7687` (started by Docker Compose)
- PostgreSQL at `postgres:5432` (started by Docker Compose)
- Redis at `redis:6379` (started by Docker Compose)
- Auth service at `http://auth-service:8000`

---

## Step 2 — Start the Stack

From the `oraclous-data-studio` directory:

```bash
docker compose up -d
```

This starts:
| Service | Port | Purpose |
|---|---|---|
| `auth-service` | 8000 | Authentication and token issuance |
| `oraclous-core-service` | 8001 | Core orchestration |
| `credential-broker-service` | 8002 | LLM credential management |
| `knowledge-graph-builder` | 8003 | **This service — the API you'll use** |
| `neo4j` | 7474, 7687 | Graph database |
| `postgres` | 5432 | Metadata and job tracking |
| `redis` | 6379 | Background job queue |

Wait for all services to become healthy (takes ~60 seconds for Neo4j to initialize):

```bash
docker compose ps
```

All services should show `healthy` or `running`. Neo4j takes the longest — it's ready when `neo4j_llm_graph` shows `healthy`.

---

## Step 3 — Get an Auth Token

All API endpoints require a Bearer token. Obtain one from the auth service:

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'
```

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

Export the token for use in subsequent commands:

```bash
export TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

> **Note:** Token format and auth endpoint paths may vary depending on your auth-service configuration. See the auth-service documentation for details.

---

## Step 4 — Verify the Service is Running

Check that the knowledge graph builder is healthy:

```bash
curl http://localhost:8003/api/v1/health
```

```json
{
  "status": "healthy",
  "service": "knowledge-graph-builder",
  "version": "1.0.0",
  "timestamp": "2026-04-07T10:00:00Z",
  "dependencies": {
    "neo4j": {"status": "healthy"},
    "postgres": {"status": "healthy"}
  }
}
```

If `status` is `degraded`, check that Neo4j and PostgreSQL containers are fully started.

---

## Step 5 — Create Your First Knowledge Graph

A knowledge graph is a named container for entities and relationships extracted from your documents.

```bash
curl -X POST http://localhost:8003/api/v1/graphs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My First Graph",
    "description": "A test knowledge graph for the quickstart guide"
  }'
```

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "My First Graph",
  "description": "A test knowledge graph for the quickstart guide",
  "user_id": "user-uuid-here",
  "node_count": 0,
  "relationship_count": 0,
  "status": "active",
  "has_instructions": false,
  "created_at": "2026-04-07T10:01:00Z",
  "updated_at": "2026-04-07T10:01:00Z"
}
```

Save your graph ID:

```bash
export GRAPH_ID="a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

---

## Step 6 — Ingest Your First Document

Send text content to the graph. The service will extract entities and relationships using LLMs and store them in Neo4j.

```bash
curl -X POST "http://localhost:8003/api/v1/graphs/$GRAPH_ID/ingest" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "TechNova Corporation was founded in 2015 by Dr. Sarah Chen and Marcus Webb. The company is headquartered in San Francisco and specializes in AI infrastructure. In 2023, TechNova acquired DataStream Inc., a data pipeline startup based in Austin. Marcus Webb serves as CEO, while Dr. Sarah Chen leads research as CTO.",
    "source_type": "text"
  }'
```

```json
{
  "id": "job-uuid-here",
  "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "progress": 0,
  "source_type": "text",
  "extracted_entities": 0,
  "extracted_relationships": 0,
  "created_at": "2026-04-07T10:02:00Z"
}
```

Ingestion runs as a background job. Poll the job status until it completes:

```bash
export JOB_ID="job-uuid-here"

curl "http://localhost:8003/api/v1/graphs/$GRAPH_ID/jobs/$JOB_ID" \
  -H "Authorization: Bearer $TOKEN"
```

When `status` becomes `completed`, your graph will contain extracted entities like `TechNova Corporation`, `Dr. Sarah Chen`, `Marcus Webb`, `DataStream Inc.` — and relationships like `FOUNDED_BY`, `ACQUIRED`, `WORKS_FOR`.

---

## Step 7 — Chat With Your Graph

Now ask a natural language question about your data:

```bash
curl -X POST http://localhost:8003/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"Who founded TechNova Corporation?\",
    \"graph_id\": \"$GRAPH_ID\",
    \"mode\": \"enhanced\"
  }"
```

```json
{
  "answer": "TechNova Corporation was founded by Dr. Sarah Chen and Marcus Webb in 2015.",
  "query": "Who founded TechNova Corporation?",
  "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "success": true,
  "mode": "enhanced",
  "retriever_type": "vector_cypher",
  "is_grounded": true,
  "confidence": 0.92,
  "sources": [
    {
      "relevance_score": 0.95,
      "content": "TechNova Corporation was founded in 2015 by Dr. Sarah Chen and Marcus Webb.",
      "entities": ["TechNova Corporation", "Dr. Sarah Chen", "Marcus Webb"]
    }
  ]
}
```

The `is_grounded: true` field confirms the answer came from your graph data, not LLM hallucination.

---

## Understanding Retriever Modes

The `mode` parameter controls how the service searches your knowledge graph:

| Mode | Description | Best For |
|---|---|---|
| `simple` | Vector similarity search | Quick factual lookups |
| `enhanced` | Vector search + graph traversal (default) | Most questions — balanced speed and context |
| `hybrid` | Vector + full-text search | Keyword-heavy queries |
| `hybrid_plus` | Hybrid search + graph traversal | Complex analytical questions |
| `natural` | Natural language → Cypher query | Precise graph analytics |

Start with `enhanced` (the default). Switch to `hybrid_plus` for research-heavy queries or `natural` when you need precise relationship counts and graph analytics.

---

## Next Steps

- **[Ingestion Guide](../user-guide/ingestion.md)** — Supported document types, ontologies, and extraction tuning
- **[Chat Guide](../user-guide/chat.md)** — Conversation history, streaming responses, and retriever configuration
- **[API Reference](../api-reference/graphs.md)** — Complete endpoint documentation with all parameters
- **[Graph Instructions](../user-guide/graphs.md)** — Set domain-specific extraction rules per graph

---

## Troubleshooting

**`503 Service Unavailable` on graph creation**
Neo4j is not yet ready. Check `docker compose ps` and wait for the neo4j container to show `healthy`.

**Ingestion job stuck in `pending`**
The background worker (`knowledge-graph-worker`) may not be running. Check:
```bash
docker compose logs knowledge-graph-worker
```

**`401 Unauthorized` on API calls**
Your token may have expired. Re-authenticate to get a fresh token.

**Ingestion job fails with LLM error**
Verify your `OPENAI_API_KEY` is set correctly in `knowledge-graph-builder/.env` and the container was restarted after the change.
