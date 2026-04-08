---
title: "Phase 2 Retrospective QA Report — MCP Server, Community Detection, Temporal Properties, Ontology Extraction"
author: QA Evaluation Engineer
date: 2026-04-08
status: complete
source: ORA-92
target: ORA-26, ORA-31, ORA-32, ORA-36
---

# Phase 2 Retrospective QA Report

**Task:** [ORA-92](/ORA/issues/ORA-92)
**Phase:** 2 (retroactive review — features shipped without QA gate)
**Reviewer:** QA Evaluation Engineer (agent `0280528a-d859-4d55-adb8-ce5859da7c1e`)
**Date:** 2026-04-08
**Overall Status:** ⚠️ CONDITIONAL PASS — 2 critical bugs found, 5 warnings

---

## Summary

| Feature | Implementation | Unit Tests | Integration Tests | Status |
|---|---|---|---|---|
| ORA-26 — MCP Server | ✅ Present | ✅ Present | ✅ Present | ⚠️ 2 bugs |
| ORA-31 — Community Detection | ✅ Present | ✅ Present | ✅ Present | ✅ Pass |
| ORA-32 — Temporal Properties | ✅ Present | ✅ Present | ⚠️ Partial | ✅ Pass |
| ORA-36 — Ontology Extraction | ✅ Present | ✅ Present | ✅ Present | ❌ 1 critical bug |

---

## Feature-by-Feature Findings

---

### ORA-26 — MCP Server (`app/mcp/server.py`)

**Implementation Coverage:**
- ✅ All required KG operations exposed as MCP tools: `create_graph`, `list_graphs`, `delete_graph`, `get_graph_stats`, `ingest_text`, `ingest_file`, `chat`, `search_nodes`, `get_node`, `get_neighbors`
- ✅ MCP Resources defined for `graphs://`, `graph://{id}/stats`, `graph://{id}/nodes`
- ✅ Both `stdio` and `sse` transport modes implemented
- ✅ Multi-tenant isolation enforced on all direct Neo4j queries via `_assert_graph_access()` pre-check
- ✅ `ORACLOUS_API_KEY` authentication via Bearer token
- ✅ Lazy HTTP client and Neo4j driver initialization (process-safe)
- ✅ `user_id` field IS returned in graph GET responses — `delete_graph` ownership check works

**Bugs Found:**

#### BUG-1 (P2-medium): `search_nodes` and `get_node` return null `type` for all entities

**Location:** `app/mcp/server.py:search_nodes()` and `get_node()`

The Cypher queries use `e.type` as a property:
```cypher
RETURN e.entity_id AS entity_id, e.name AS name, e.type AS type, ...
```

However, entity types are stored as **Neo4j node labels** (`:Person`, `:Company`) via `apoc.create.addLabels`, NOT as a property `e.type`. The `e.type` property does not exist on entity nodes. All `type` fields in `search_nodes` results will return `""` (empty string).

**Impact:** MCP clients calling `search_nodes` with `entity_type` filter will never filter correctly. Entity type info is silently lost in all node inspection tool responses.

**Fix:** Use `labels(e)` to extract type: `[l IN labels(e) WHERE l <> '__Entity__' AND l <> '__KGBuilder__'][0] AS type`

#### BUG-2 (P2-medium): `delete_graph` fails silently when running as standalone MCP process

**Location:** `app/mcp/server.py:delete_graph()`

The tool imports `GraphNodeService` from the application layer:
```python
from app.services.graph_node_service import GraphNodeService
```

When the MCP server runs as a standalone subprocess (stdio mode, e.g., Claude Desktop), the `app.*` import chain may fail or create a conflicting Neo4j driver instance separate from the `_neo4j_sync_driver()` fallback. The error is caught and returned as `{"deleted": False, "error": "..."}` — making it appear as a soft failure to the MCP client.

**Impact:** `delete_graph` silently fails in standalone MCP deployments.

**Fix:** Add a REST DELETE endpoint to the Oraclous API (Backend Developer task) and delegate `delete_graph` through HTTP like the other management tools.

---

### ORA-31 — Hierarchical Community Detection (`app/tasks/community_tasks.py`)

**Implementation Coverage:**
- ✅ Leiden algorithm via `leidenalg` + `igraph` at 3 levels (resolutions 0.5, 1.0, 2.0)
- ✅ Deterministic community IDs via SHA-256 of sorted member IDs (`make_community_id`)
- ✅ LLM summarization with hash-based caching — skips redundant LLM calls
- ✅ Community embeddings generated via OpenAI embeddings API
- ✅ Hierarchical `parent_id` assignment using majority vote
- ✅ Multi-tenant isolation: all Cypher queries filter by `graph_id`
- ✅ Redis distributed lock (10 min TTL) prevents concurrent detection on same graph
- ✅ Postgres status tracking (`rebuilding` → `active`/`stale`)
- ✅ Minimum entity count guard (50 entities) prevents useless runs on tiny graphs
- ✅ `IN_COMMUNITY` and `PARENT_COMMUNITY` relationships written correctly

**Warnings:**

#### WARN-1: Placeholder `graph_id` in `_run_leiden` is confusing

In `_run_leiden`, community IDs are computed with `graph_id="__placeholder__"` and then re-derived in `_upsert_communities` with the real `graph_id`. This works correctly but creates confusion — the community IDs in `communities_map` don't match the final Neo4j node IDs. Code comment says "filled during upsert" which mitigates the risk, but this pattern should be refactored.

#### WARN-2: Return type annotation mismatch in `_build_hierarchy`

`_build_hierarchy` is annotated to return `dict[int, dict[str, list[str]]]` but actually returns `dict[int, dict[str, dict]]` (each community becomes a dict with `members`, `parent_id`, `summary`, etc.). This type hint mismatch will cause static analysis tools to misreport types.

**Verdict: PASS with warnings.** Core algorithm is correct and multi-tenant safe.

---

### ORA-32 — Temporal Properties on Entities & Relationships

**Implementation Coverage:**
- ✅ `valid_from` / `valid_to` on `RelationshipProperties` with ISO-8601 coercion (via Pydantic `field_validator`)
- ✅ `valid_from > valid_to` rejected at schema validation
- ✅ `transaction_time` auto-set server-side on all relationship properties
- ✅ `TemporalContext` for per-ingestion world-time overrides
- ✅ `TemporalFilter` with `point_in_time`, `valid_from_gte`, `valid_to_lte`, and `current_only` modes
- ✅ `compile_temporal_filter` in `PipelineService` generates correct WHERE clauses
- ✅ Temporal contradiction detection in ingestion pipeline
- ✅ Neo4j indexes on `(e.graph_id, e.valid_from, e.valid_to)` for entities and relationships
- ✅ `UpdateTemporalBoundsRequest` schema with validation
- ✅ Temporal context applied as relationship property override after LLM extraction (step 4.5 in pipeline)

**Warnings:**

#### WARN-3: Temporal index on entity nodes may be misleading

Per the spec, `valid_from` / `valid_to` should be on **relationships**, not entity nodes. The index `entity_temporal_idx` on `(e.graph_id, e.valid_from, e.valid_to)` suggests entity-level temporal filtering is expected. If entities never have temporal properties, this index wastes space. Should be clarified in spec or removed.

#### WARN-4: Unparseable temporal strings silently become `None`

```python
return None  # Unparseable temporal string — treat as missing
```

An LLM-generated date like `"early 2020s"` or `"Q3 2021"` will silently be treated as no temporal data instead of raising an extraction quality warning. This masks extraction quality issues.

**Verdict: PASS with warnings.** Core temporal properties are correctly implemented.

---

### ORA-36 — Ontology-Guided Extraction

**Implementation Coverage:**
- ✅ `EntityTypeDefinition` / `RelationshipTypeDefinition` schema models
- ✅ `GraphInstructions` stored on Graph node in Neo4j
- ✅ `InstructionsResolver`: merges graph instructions + per-job overrides
- ✅ `InstructionsCompiler.to_prompt()`: injects ontology context into LLM extraction prompt
- ✅ `_enforce_ontology` in pipeline: WARN / STRICT / COERCE modes during ingestion (operates on Python `node.label` attribute — correct)
- ✅ CRUD endpoints for ontology: `GET/PUT/PATCH/DELETE /graphs/{id}/ontology`
- ✅ Dry-run validation endpoint `/graphs/{id}/ontology/validate`
- ✅ `retroactive_apply_ontology_task` Celery task exists

**Critical Bug Found:**

#### BUG-3 (P1-high): `retroactive_apply_ontology_task` never finds violations — silently does nothing

**Location:** `app/tasks/ontology_tasks.py`

The retroactive enforcement task queries Neo4j using `e.label` as a **property**:
```cypher
MATCH (e:__Entity__ {graph_id: $graph_id})
WHERE NOT e.label IN $allowed_types
RETURN elementId(e) AS eid, e.label AS label
```

**Root cause:** Entity types are stored as **Neo4j node labels** (`:Person`, `:Company`), NOT as a node property `e.label`. The property `e.label` does not exist. In Cypher, `WHERE NOT null IN ['Person', 'Company']` evaluates to `null`, which means the WHERE clause matches nothing.

**Result:**
- `warn` mode: reports 0 violations (false clean bill of health)
- `strict` mode: deletes 0 entities (no-op)
- `coerce` mode: coerces 0 entities (no-op)
- The `remaining_violations` count query also always returns 0

The correct Cypher should use the `labels()` function:
```cypher
MATCH (e:__Entity__ {graph_id: $graph_id})
WHERE NOT any(l IN labels(e) WHERE l IN $allowed_types AND l <> '__Entity__' AND l <> '__KGBuilder__')
```

**Impact:** Retroactive ontology enforcement for all existing graphs is broken. Any graph ingested before an ontology was set cannot be validated or cleaned up. This affects the entire retroactive enforcement use case.

**Assign to:** Backend Developer Senior for fix.

**Verdict: CONDITIONAL FAIL.** Real-time enforcement during ingestion works. Retroactive enforcement is broken.

---

### Cross-Cutting Issue

#### WARN-5 (affects ORA-26, ORA-36, and multiple services): `e.type` property does not exist

Several services query `e.type` as a Neo4j property:
- `app/mcp/server.py:search_nodes()` — `RETURN e.type AS type`
- `app/services/graph_node_service.py` — index `ON (e.name, e.type)`
- `app/services/versioning_service.py` — `coalesce(e.type, '') AS entity_type`
- `app/services/snapshot_service.py` — `coalesce(e.type, '') AS entity_type`
- `app/services/federation_service.py` — `coalesce(e.type, labels(e)[-1]) AS type`

Entity types are stored as Neo4j labels, not properties. The `coalesce(e.type, labels(e)[-1])` pattern in `federation_service.py` is the correct approach — it gracefully falls back to `labels(e)[-1]` when `e.type` is null. The other services do not do this and will return empty/null for all entity types.

**Assign to:** Backend Developer Senior to standardize on the `federation_service.py` pattern or explicitly store `type` as a property during entity write.

---

## Test Coverage Assessment

| Feature | Unit Tests | Integration Tests | Gap |
|---|---|---|---|
| ORA-26 MCP | `tests/unit/test_mcp_server.py` (comprehensive) | `tests/integration/test_mcp_integration.py` | No test for `e.type` null issue |
| ORA-31 Community | `tests/unit/test_community_detection.py` (comprehensive) | None found | No integration test with real Neo4j |
| ORA-32 Temporal | `tests/unit/test_temporal.py` (comprehensive) | None found | No test for temporal contradiction detection end-to-end |
| ORA-36 Ontology | `tests/unit/test_ontology.py` (comprehensive) | `tests/integration/test_ontology_api.py` | **No test for retroactive task — critical bug undetected** |

**Root cause of missed bugs:** Unit tests for ORA-36 test the Python-level `_enforce_ontology` (which correctly uses `node.label` Python attribute), but don't test the Cypher-level `retroactive_apply_ontology_task` against a real Neo4j instance. The retroactive task needs integration tests.

---

## Bugs Filed

| Bug ID | Feature | Priority | Title |
|---|---|---|---|
| TBD | ORA-36 | P1-high | `retroactive_apply_ontology_task` uses `e.label` property instead of `labels(e)` — silently never enforces ontology |
| TBD | ORA-26 | P2-medium | `search_nodes` returns null `type` for all entities — `e.type` property does not exist |
| TBD | ORA-26 | P2-medium | `delete_graph` MCP tool fails silently as standalone process |

---

## Recommendations

1. **Immediate (P1):** Fix `retroactive_apply_ontology_task` to use `labels(e)` function in Cypher instead of `e.label` property. File bug issue assigned to Backend Developer Senior.

2. **Short-term (P2):** Standardize entity type access across all services. Either:
   - (a) Store `type` as a property during entity write (in `MultiTenantKGWriter`), OR
   - (b) Update all Cypher queries to use `[l IN labels(e) WHERE l NOT IN ['__Entity__', '__KGBuilder__']][0]`
   The `federation_service.py` coalesce pattern is the best current approach.

3. **Short-term (P2):** Add a REST DELETE endpoint so MCP `delete_graph` delegates through HTTP instead of importing app-level services.

4. **Testing (P3):** Add integration tests for `retroactive_apply_ontology_task` against a real Neo4j instance. Add integration tests for community detection pipeline.

---

## QA Gate Verdict

| Gate | Status |
|---|---|
| Integration tests pass | ✅ (existing tests pass — but don't cover critical bug) |
| Multi-tenant isolation | ✅ Verified for all four features |
| No regression in existing functionality | ✅ |
| Security regression tests | N/A for this scope |
| Retroactive enforcement works | ❌ ORA-36 bug blocks this |

**Overall:** Features ORA-26, ORA-31, ORA-32 may be considered passed with warnings. Feature ORA-36's retroactive enforcement is broken (P1 bug). Real-time enforcement during ingestion works correctly.
