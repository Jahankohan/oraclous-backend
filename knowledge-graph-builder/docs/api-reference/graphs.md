# API Reference: Graph Management

Base URL: `http://localhost:8003/api/v1`

All endpoints require `Authorization: Bearer <token>` header.

---

## Graphs

### Create a Knowledge Graph

Creates a new knowledge graph in Neo4j. Each graph belongs to the authenticated user and is isolated from other users' graphs.

```
POST /graphs
```

**Request Body**

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Graph name (1–255 characters) |
| `description` | string | No | Human-readable description (max 1000 characters) |
| `schema_config` | object | No | Optional schema configuration |

```bash
curl -X POST http://localhost:8003/api/v1/graphs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Company Knowledge Base",
    "description": "Internal company org chart and relationships",
    "schema_config": {}
  }'
```

```python
import httpx

response = httpx.post(
    "http://localhost:8003/api/v1/graphs",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "name": "Company Knowledge Base",
        "description": "Internal company org chart and relationships"
    }
)
graph = response.json()
graph_id = graph["id"]
```

**Response** `201 Created`

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "name": "Company Knowledge Base",
  "description": "Internal company org chart and relationships",
  "user_id": "user-uuid-here",
  "schema_config": {},
  "node_count": 0,
  "relationship_count": 0,
  "status": "active",
  "has_instructions": false,
  "instructions_version": null,
  "last_optimized": null,
  "optimization_count": 0,
  "last_optimization_type": null,
  "created_at": "2026-04-07T10:00:00Z",
  "updated_at": "2026-04-07T10:00:00Z"
}
```

**Errors**

| Status | Detail | Cause |
|---|---|---|
| `401` | Unauthorized | Missing or invalid Bearer token |
| `422` | Validation Error | `name` empty or exceeds 255 characters |
| `503` | Neo4j connection not available | Neo4j service is not running |

---

### List Knowledge Graphs

Returns all graphs owned by the authenticated user.

```
GET /graphs
```

```bash
curl http://localhost:8003/api/v1/graphs \
  -H "Authorization: Bearer $TOKEN"
```

```python
response = httpx.get(
    "http://localhost:8003/api/v1/graphs",
    headers={"Authorization": f"Bearer {token}"}
)
graphs = response.json()
```

**Response** `200 OK`

```json
[
  {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "name": "Company Knowledge Base",
    "description": "Internal company org chart and relationships",
    "user_id": "user-uuid-here",
    "node_count": 142,
    "relationship_count": 87,
    "status": "active",
    "has_instructions": true,
    "instructions_version": 2,
    "created_at": "2026-04-07T10:00:00Z",
    "updated_at": "2026-04-07T10:05:00Z"
  }
]
```

---

### Get a Knowledge Graph

Returns details for a specific graph. Returns `403` if the graph belongs to another user.

```
GET /graphs/{graph_id}
```

**Path Parameters**

| Parameter | Type | Description |
|---|---|---|
| `graph_id` | UUID | Graph identifier |

```bash
curl "http://localhost:8003/api/v1/graphs/$GRAPH_ID" \
  -H "Authorization: Bearer $TOKEN"
```

**Response** `200 OK` — same shape as the Create response.

**Errors**

| Status | Detail | Cause |
|---|---|---|
| `403` | Access denied | Graph belongs to another user |
| `404` | Graph not found | No graph with that ID exists |

---

### Update a Knowledge Graph

Updates the name or description of a graph. Only fields included in the request are updated.

```
PUT /graphs/{graph_id}
```

**Request Body** (all fields optional)

| Field | Type | Description |
|---|---|---|
| `name` | string | New name (1–255 characters) |
| `description` | string | New description (max 1000 characters) |
| `schema_config` | object | Updated schema configuration |

```bash
curl -X PUT "http://localhost:8003/api/v1/graphs/$GRAPH_ID" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Graph Name",
    "description": "Updated description"
  }'
```

**Response** `200 OK` — updated graph object.

---

### Delete a Knowledge Graph

> **Note:** This endpoint is not yet implemented. Use Neo4j directly to delete graph data if needed.

---

## Ingestion

### Ingest a Document

Submits content for entity and relationship extraction. Runs as a background job — the response contains a `job_id` you poll for status.

```
POST /graphs/{graph_id}/ingest
```

**Request Body**

| Field | Type | Required | Description |
|---|---|---|---|
| `content` | string | Yes | Text content to ingest (min 10 characters) |
| `source_type` | string | No | Content type: `text`, `pdf`, `url`, `api` (default: `text`) |
| `overrides` | object | No | Per-job extraction overrides (see below) |
| `evolution_mode` | string | No | Schema evolution: `strict`, `guided`, `permissive` (default: `guided`) |
| `max_entities` | integer | No | Max entity types allowed (default: 20) |
| `max_relationships` | integer | No | Max relationship types allowed (default: 15) |
| `enforce_relationship_properties` | boolean | No | Validate contextual properties are on edges (default: `true`) |

**Overrides Object**

Per-job settings that supplement (but do not replace) graph-level instructions:

| Field | Type | Description |
|---|---|---|
| `additional_focus` | string | One-time focus hint for this document |
| `override_density` | string | Override extraction density: `sparse`, `balanced`, `dense` |
| `extra_entity_types` | array | Additional entity types beyond graph defaults |
| `schema_evolution_hint` | string | How schema should evolve from this document |

```bash
curl -X POST "http://localhost:8003/api/v1/graphs/$GRAPH_ID/ingest" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "TechNova Corporation was founded in 2015 by Dr. Sarah Chen and Marcus Webb...",
    "source_type": "text",
    "overrides": {
      "additional_focus": "Focus on executive roles and reporting relationships",
      "override_density": "dense"
    }
  }'
```

```python
response = httpx.post(
    f"http://localhost:8003/api/v1/graphs/{graph_id}/ingest",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "content": open("document.txt").read(),
        "source_type": "text",
        "overrides": {
            "additional_focus": "Focus on product names and features"
        }
    }
)
job = response.json()
job_id = job["id"]
```

**Response** `201 Created`

```json
{
  "id": "job-uuid-here",
  "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "progress": 0,
  "source_type": "text",
  "extracted_entities": 0,
  "extracted_relationships": 0,
  "processed_chunks": 0,
  "created_at": "2026-04-07T10:02:00Z"
}
```

**Deprecated Field**

The `instructions` string field (top-level) is deprecated. Use `overrides.additional_focus` instead.

---

### List Ingestion Jobs

Returns all ingestion jobs for a graph, ordered by creation date (newest first).

```
GET /graphs/{graph_id}/jobs
```

```bash
curl "http://localhost:8003/api/v1/graphs/$GRAPH_ID/jobs" \
  -H "Authorization: Bearer $TOKEN"
```

**Response** `200 OK`

```json
[
  {
    "id": "job-uuid-here",
    "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "status": "completed",
    "progress": 100,
    "source_type": "text",
    "extracted_entities": 8,
    "extracted_relationships": 12,
    "processed_chunks": 3,
    "similarity_relationships": 4,
    "communities_detected": 2,
    "property_violations_detected": 0,
    "property_violations_migrated": 0,
    "created_at": "2026-04-07T10:02:00Z",
    "started_at": "2026-04-07T10:02:01Z",
    "completed_at": "2026-04-07T10:02:45Z"
  }
]
```

**Job Status Values**

| Status | Meaning |
|---|---|
| `pending` | Job queued, waiting for background worker |
| `running` | Extraction in progress |
| `completed` | Successfully finished |
| `failed` | Extraction failed — check `error_message` |

---

### Get Ingestion Job Status

Polls the status of a specific ingestion job.

```
GET /graphs/{graph_id}/jobs/{job_id}
```

```bash
curl "http://localhost:8003/api/v1/graphs/$GRAPH_ID/jobs/$JOB_ID" \
  -H "Authorization: Bearer $TOKEN"
```

**Response** `200 OK` — same shape as a single job in the list response, with full detail including `error_message` and timestamps.

**Errors**

| Status | Detail | Cause |
|---|---|---|
| `404` | Ingestion job not found | Job ID doesn't exist or belongs to a different graph |

---

## Graph Instructions

Set domain-specific extraction rules that apply to all future ingestion jobs on a graph.

### Set Graph Instructions

Replaces any existing instructions with the new set. Instructions are versioned — each update increments the version number.

```
PUT /graphs/{graph_id}/instructions
```

**Request Body**

| Field | Type | Required | Description |
|---|---|---|---|
| `domain` | string | No | Domain hint: `"HR org chart"`, `"pharmaceutical research"`, etc. |
| `extraction_density` | string | No | `sparse`, `balanced`, `dense` (default: `balanced`) |
| `entity_types` | array | No | Preferred entity types. When set: ontology-guided mode. When absent: free-form. |
| `relationship_types` | array | No | Preferred relationship types with edge-property rules |
| `edge_property_fields` | array | No | Property names that must always be on relationships, not nodes |
| `focus_areas` | array | No | Free-text hints about what to emphasize |
| `ignore_patterns` | array | No | Entity names or patterns to skip |
| `language` | string | No | ISO 639-1 language code (default: `en`) |
| `custom_prompt_suffix` | string | No | Text appended verbatim to the extraction prompt (max 2000 chars) |

```bash
curl -X PUT "http://localhost:8003/api/v1/graphs/$GRAPH_ID/instructions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "Corporate org chart",
    "extraction_density": "dense",
    "entity_types": [
      {
        "name": "Person",
        "description": "An individual employee or executive",
        "examples": ["Dr. Sarah Chen", "Marcus Webb"]
      },
      {
        "name": "Company",
        "description": "An organization or business entity",
        "examples": ["TechNova Corporation", "DataStream Inc."]
      },
      {
        "name": "Department",
        "description": "A division or team within a company"
      }
    ],
    "relationship_types": [
      {
        "name": "WORKS_FOR",
        "source_type": "Person",
        "target_type": "Company",
        "store_as_edge_property": ["job_title", "start_date"]
      },
      {
        "name": "REPORTS_TO",
        "source_type": "Person",
        "target_type": "Person"
      }
    ],
    "edge_property_fields": ["job_title", "role", "start_date"],
    "focus_areas": ["Executive leadership", "Reporting structure"],
    "ignore_patterns": ["Inc.", "LLC", "Ltd."]
  }'
```

**Response** `200 OK`

```json
{
  "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "instructions": {
    "domain": "Corporate org chart",
    "extraction_density": "dense",
    "entity_types": [...],
    "relationship_types": [...],
    "edge_property_fields": ["job_title", "role", "start_date"],
    "focus_areas": ["Executive leadership", "Reporting structure"],
    "ignore_patterns": ["Inc.", "LLC", "Ltd."],
    "language": "en"
  },
  "version": 1,
  "updated_at": "2026-04-07T10:03:00Z"
}
```

---

### Get Graph Instructions

```
GET /graphs/{graph_id}/instructions
```

```bash
curl "http://localhost:8003/api/v1/graphs/$GRAPH_ID/instructions" \
  -H "Authorization: Bearer $TOKEN"
```

**Response** `200 OK` — same shape as the Set Instructions response.

**Errors**

| Status | Detail | Cause |
|---|---|---|
| `404` | No instructions configured for this graph | No instructions have been set yet |

---

### Delete Graph Instructions

Removes all instructions and reverts the graph to free-form extraction mode.

```
DELETE /graphs/{graph_id}/instructions
```

```bash
curl -X DELETE "http://localhost:8003/api/v1/graphs/$GRAPH_ID/instructions" \
  -H "Authorization: Bearer $TOKEN"
```

**Response** `204 No Content`

---

## Migration

### Migrate Node Properties to Relationships

Runs a 3-phase migration that moves contextual properties (like `job_title`, `role`) from entity nodes onto relationship edges where they semantically belong. Safe to run multiple times — phases 1 and 2 are idempotent.

```
POST /graphs/{graph_id}/migrate-properties
```

```bash
curl -X POST "http://localhost:8003/api/v1/graphs/$GRAPH_ID/migrate-properties" \
  -H "Authorization: Bearer $TOKEN"
```

**Response** `200 OK`

```json
{
  "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "phase1_nodes_scanned": 142,
  "phase2_properties_moved": 38,
  "phase3_orphans_logged": 2,
  "status": "completed"
}
```

---

## Health Check

### Service Health

Returns the health status of the service and its dependencies (Neo4j and PostgreSQL). Does not require authentication.

```
GET /health
```

```bash
curl http://localhost:8003/api/v1/health
```

**Response** `200 OK`

```json
{
  "status": "healthy",
  "service": "knowledge-graph-builder",
  "version": "1.0.0",
  "timestamp": "2026-04-07T10:00:00Z",
  "dependencies": {
    "neo4j": {
      "status": "healthy",
      "latency_ms": 12
    },
    "postgres": {
      "status": "healthy",
      "latency_ms": 3
    }
  }
}
```

**Status Values**

| Status | Meaning |
|---|---|
| `healthy` | All dependencies are reachable |
| `degraded` | One or more dependencies are unavailable |
