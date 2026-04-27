# API Reference: Schema Management

Base URL: `http://localhost:8003/api/v1`

> **Authentication note:** Schema endpoints currently do not require authentication (auth is pending implementation). They will be secured in a future release.

The schema service extracts and caches the live Neo4j schema for a graph — the set of node labels, relationship types, and properties that currently exist in the database. Schema data is used internally by the `natural` (Text2Cypher) chat mode to generate accurate Cypher queries.

---

## Get Schema Info

Returns the current schema for a graph, including all node labels and relationship types with sample counts and properties.

```
GET /schema/info/{graph_id}
```

**Path Parameters**

| Parameter | Type | Description |
|---|---|---|
| `graph_id` | string | Knowledge graph UUID |

```bash
curl "http://localhost:8003/api/v1/schema/info/$GRAPH_ID"
```

```python
import httpx

response = httpx.get(f"http://localhost:8003/api/v1/schema/info/{graph_id}")
schema = response.json()
```

**Response** `200 OK`

```json
{
  "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "schema_version": "v1.2.3",
  "last_updated": "2026-04-07T10:05:00Z",
  "nodes": {
    "Person": {
      "sample_count": 12,
      "property_count": 3,
      "properties": ["name", "description", "ingested_at"]
    },
    "Company": {
      "sample_count": 5,
      "property_count": 2,
      "properties": ["name", "description"]
    },
    "__Entity__": {
      "sample_count": 142,
      "property_count": 4,
      "properties": ["id", "name", "description", "embedding"]
    }
  },
  "relationships": {
    "WORKS_FOR": {
      "sample_count": 8,
      "property_count": 3,
      "start_labels": ["Person"],
      "end_labels": ["Company"]
    },
    "ACQUIRED": {
      "sample_count": 2,
      "property_count": 2,
      "start_labels": ["Company"],
      "end_labels": ["Company"]
    }
  },
  "constraints": 3,
  "indexes": 7
}
```

> **Note:** Each node entry shows up to 5 properties. Each relationship entry shows up to 3 source and target label types.

---

## Get Text2Cypher Schema

Returns the schema formatted as a string for the Text2Cypher retriever. This is the exact format passed to the LLM when generating Cypher from natural language in `natural` mode.

```
GET /schema/text2cypher/{graph_id}
```

**Query Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `force_refresh` | boolean | `false` | Bypass cache and re-extract schema from Neo4j |

```bash
curl "http://localhost:8003/api/v1/schema/text2cypher/$GRAPH_ID?force_refresh=false"
```

**Response** `200 OK`

```json
{
  "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "schema_version": "v1.2.3",
  "formatted_schema": "Node labels: Person (properties: name, description), Company (properties: name, description).\nRelationship types: WORKS_FOR (Person -> Company), ACQUIRED (Company -> Company), FOUNDED_BY (Company -> Person).",
  "last_updated": "2026-04-07T10:05:00Z"
}
```

---

## Refresh Schema Cache

Forces a fresh schema extraction from Neo4j and updates the cache. Use this after significant ingestion runs when the Text2Cypher schema feels stale.

```
POST /schema/refresh
```

**Request Body**

| Field | Type | Required | Description |
|---|---|---|---|
| `graph_id` | string | Yes | Knowledge graph UUID |
| `force_refresh` | boolean | No | Force refresh even if cache is fresh (default: `false`) |

```bash
curl -X POST http://localhost:8003/api/v1/schema/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "force_refresh": true
  }'
```

**Response** `200 OK` — Updated `SchemaInfo` object (same shape as `GET /schema/info/{graph_id}`).

---

## Clear Schema Cache

### Clear Cache for One Graph

```
DELETE /schema/cache/{graph_id}
```

```bash
curl -X DELETE "http://localhost:8003/api/v1/schema/cache/$GRAPH_ID"
```

**Response** `200 OK`

```json
{
  "message": "Schema cache cleared for graph a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

---

### Clear All Schema Caches

```
DELETE /schema/cache
```

```bash
curl -X DELETE http://localhost:8003/api/v1/schema/cache
```

**Response** `200 OK`

```json
{
  "message": "All schema caches cleared"
}
```

---

## Get Cache Status

Returns metadata about all currently cached schemas — useful for debugging staleness issues.

```
GET /schema/cache/status
```

```bash
curl http://localhost:8003/api/v1/schema/cache/status
```

**Response** `200 OK`

```json
{
  "a1b2c3d4-e5f6-7890-abcd-ef1234567890": {
    "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "schema_version": "v1.2.3",
    "last_updated": "2026-04-07T10:05:00Z",
    "node_count": 5,
    "relationship_count": 8,
    "age_minutes": 12.4
  }
}
```

---

## Schema Service Health

```
GET /schema/health
```

```bash
curl http://localhost:8003/api/v1/schema/health
```

**Response** `200 OK`

```json
{
  "status": "healthy",
  "service": "schema_manager",
  "cached_schemas": 3,
  "total_cache_size_mb": 0.04
}
```
