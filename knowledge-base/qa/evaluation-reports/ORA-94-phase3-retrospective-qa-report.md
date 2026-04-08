---
title: "Phase 3 Retrospective QA Report — ReBAC, Incremental KG Updates, Cross-Graph Federation & Graph Versioning"
author: QA Evaluation Engineer
date: 2026-04-08
status: complete
source: ORA-94
target: ORA-52, ORA-53, ORA-54, ORA-55, ORA-56, ORA-60, ORA-61, ORA-62, ORA-63
---

# Phase 3 Retrospective QA Report

**Task:** [ORA-94](/ORA/issues/ORA-94)
**Phase:** 3 (retroactive review — features shipped without QA gate)
**Reviewer:** QA Evaluation Engineer (agent `0280528a-d859-4d55-adb8-ce5859da7c1e`)
**Date:** 2026-04-08
**Overall Status:** ⚠️ CONDITIONAL PASS — 1 P1 critical bug, 1 P2 bug, 5 warnings

---

## Summary

| Feature | Implementation | Unit Tests | Integration Tests | Status |
|---|---|---|---|---|
| ORA-52/ORA-63 — ReBAC Phase A+B | ✅ Present | ✅ Present | ✅ Present (575 lines) | ✅ Pass |
| ORA-53/ORA-60 — Incremental KG Updates | ✅ Present | ✅ Present | ❌ None | ⚠️ Warning |
| ORA-54 — Cross-Graph Federation | ✅ Present | ⚠️ Partial | ⚠️ Partial | ❌ P1 bug |
| ORA-55 — Graph Versioning | ✅ Present | ✅ Present | ✅ Present (681 lines) | ⚠️ Warning |
| ORA-56 — OpenTelemetry | ✅ Present | ✅ Present | ✅ Present | ✅ Pass |
| ORA-61 — Cypher Injection Fix | ✅ Present | ✅ Present | ✅ Present | ✅ Pass |
| ORA-62 — JWT sub→UUID Migration | ✅ Present | ✅ Present | ⚠️ Partial | ✅ Pass |

---

## Feature-by-Feature Findings

---

### ORA-52 / ORA-63 — ReBAC Access Control (`app/services/rebac_service.py`)

**Implementation Coverage:**
- ✅ Phase A (legacy): 3-path resolution — direct `CAN_ACCESS`, team `MEMBER_OF` + `CAN_ACCESS`, org `BELONGS_TO {role:owner}` + `OWNS`
- ✅ Phase B (ORA-48): `HAS_ROLE → HAS_PERMISSION | INHERITS_FROM*0..5` traversal
- ✅ 5 built-in roles: `owner`, `admin`, `editor`, `viewer`, `restricted_viewer` with correct permission sets
- ✅ 13 system `Permission` nodes seeded idempotently on startup
- ✅ Role inheritance chain: `owner→admin→editor→viewer` with `restricted_viewer` from `owner` and `admin`
- ✅ `INHERITS_FROM` max depth 5 — prevents infinite traversal
- ✅ Redis permission cache (TTL=60s) with graceful fallback on Redis unavailability
- ✅ `is_active` soft-revoke on `HAS_ROLE` edges (no hard deletes)
- ✅ `expires_at` on `HAS_ROLE` edges — time-limited grants checked in Cypher
- ✅ `bootstrap_graph_roles` called automatically on new graph creation
- ✅ `sync_existing_data` migrates all pre-existing graphs to Phase A CAN_ACCESS (idempotent MERGE)
- ✅ `SubGraph` partition management for scoped access control
- ✅ Architecture Rule #4 enforced: every Cypher query includes `graph_id` filter
- ✅ Integration tests cover direct grant, team grant, org ownership, expired grants, and injection attempts (`test_rebac.py`, 575 lines, 24+ test functions)
- ✅ API endpoints: `GET/POST /graphs/{id}/members`, `DELETE /graphs/{id}/members/{userId}`, `GET/POST /graphs/{id}/subgraphs`

**Warnings:**

#### WARN-1: Phase A CAN_ACCESS grants invisible once Phase B roles are bootstrapped for a graph

`sync_existing_data()` creates Phase A `CAN_ACCESS` edges for existing graphs but does NOT run `bootstrap_graph_roles()`. This is correct behavior — existing graphs have no Phase B roles, so `check_graph_permission()` falls through to Phase A (because `role_check` returns `cnt=0`).

However, if an operator manually calls `bootstrap_graph_roles()` retroactively for an existing graph (e.g., during a migration), any user who holds only a Phase A `CAN_ACCESS` grant (but not a Phase B `HAS_ROLE` edge) would be **permanently denied** by Phase B (which finds roles exist for the graph and returns `authorized=False`), with no fallthrough to Phase A.

**Risk:** Low for normal operations (bootstrap only runs automatically for new graphs). Becomes a risk if a future migration script bootstraps Phase B for existing graphs without also migrating existing `CAN_ACCESS` grants to `HAS_ROLE`.

**Recommendation:** Document this transition contract clearly. Any future Phase B migration script for existing graphs MUST also call `grant_role()` for all users with existing `CAN_ACCESS` edges before bootstrap completes.

**Verdict: PASS.** ReBAC is architecturally sound and all security-critical paths are tested. Warning is a future operational risk, not a current bug.

---

### ORA-53 / ORA-60 — Incremental KG Updates (`app/services/pipeline_service.py`, `app/services/background_jobs.py`)

**Implementation Coverage:**
- ✅ `IngestMode` enum: `INCREMENTAL`, `FULL` modes
- ✅ Hash guard: `_check_document_hash_unchanged()` — SHA-256 of document bytes; returns `status=skipped` for unchanged docs, zero Neo4j writes
- ✅ Delta detection: only new chunks processed on partial re-ingestion
- ✅ Manual property preservation across re-ingestion cycles
- ✅ Stale chunk cleanup: `staleAt` timestamp set on removed chunks
- ✅ `ingest_mode` stored in `effective_instructions` on background jobs and reconstructed at Celery worker resume time
- ✅ Default mode is `INCREMENTAL` when no mode specified
- ✅ Migration backfill script: `scripts/backfill_incremental_kg.py`
- ✅ Unit tests: `tests/unit/test_incremental_kg.py` covers all 7 test criteria from spec (hash guard, idempotency, delta, manual property preservation, stale cleanup, full mode, migration backfill)

**Gaps Found:**

#### WARN-2: No integration tests for Incremental KG against real Neo4j

All 7 test criteria are verified at the unit level with mocked Neo4j (`pipeline._neo4j_client = mock_client`). No integration tests run the full `incremental` ingestion path against a real Neo4j instance.

Specific behaviors that are unit-tested but not integration-tested:
- The actual Cypher queries that check document hashes in Neo4j
- Stale chunk detection and `staleAt` marking in real graph data
- Delta merge behavior when chunk embeddings differ only slightly

**Impact:** Low risk for the hash guard (simple hash comparison), higher risk for stale chunk marking and delta detection which rely on complex Cypher logic.

**Recommendation:** Add at least 2 integration tests: (1) re-ingesting unchanged document → verify no new Chunk/Entity nodes created in Neo4j; (2) partial re-ingestion → verify stale chunks have `staleAt` set and new chunks are added.

**Verdict: CONDITIONAL PASS.** Core logic is correct and unit-tested. Integration test gap is a testing quality issue, not a functionality bug.

---

### ORA-54 — Cross-Graph Federation (`app/services/federation_service.py`)

**Implementation Coverage:**
- ✅ Fail-closed permission model: all graph IDs must pass validation or request is rejected entirely
- ✅ `federatable=true` flag required on all target graphs
- ✅ Service account path: delegates to `service_account_service.get_sa_accessible_graphs()`
- ✅ UNION ALL per-graph Cypher subqueries with graph-scoped indexes
- ✅ Over-fetch pattern for vector search (`1.5x` multiplier)
- ✅ SAME_AS entity deduplication via exact name+type matching (confidence 0.99)
- ✅ Async fire-and-forget SAME_AS edge storage in Neo4j
- ✅ 10 graph max per request enforced
- ✅ API endpoints: `POST /graphs/federate/query`, `POST /graphs/federate/vector-search`
- ✅ `coalesce(e.type, labels(e)[-1]) AS type` — correctly handles entity types as labels

**Critical Bug Found:**

#### BUG-1 (P1-critical): `_validate_and_filter` reads `g.user_id` but ReBAC stores `g.owner_user_id` — all federated queries return 403

**Location:** `app/services/federation_service.py:_validate_and_filter()` (line ~202)

The validation query reads:
```cypher
MATCH (g:Graph)
WHERE g.graph_id IN $graph_ids
RETURN g.graph_id AS graph_id,
       g.user_id AS user_id,  ← BUG: property is owner_user_id
       ...
```

However, ReBAC's `register_new_graph()` stores the owner as `g.owner_user_id`, not `g.user_id`:
```cypher
MERGE (g:Graph {graph_id: $graph_id, namespace: "__system__"})
ON CREATE SET
    g.name = $name,
    g.owner_user_id = $user_id,  ← stored here
    ...
```

**Result:** `row["user_id"]` is always `None`. The comparison `if row["user_id"] != user_id` evaluates as `None != "<any_user_id>"` = `True`, which triggers:
```python
raise FederationError("Access denied — ...", status_code=403)
```

**Every federated query by every user will return HTTP 403**, regardless of actual ownership. The feature is completely non-functional for the user path (not the service account path, which uses a different code branch).

**Fix:** Change `g.user_id AS user_id` → `g.owner_user_id AS user_id` in the MATCH query in `_validate_and_filter()`.

**Also note:** The MATCH does not filter by `namespace: "__system__"` — if any non-ReBAC `:Graph` nodes exist in Neo4j (e.g., from legacy code), they could match and have neither `user_id` nor `owner_user_id`, causing false positives.

**Assign to:** Backend Developer Senior.

**BUG-2 (P2-medium): `federated_vector_search` always passes empty query vector — vector search is non-functional**

**Location:** `app/services/federation_service.py:_execute_vector_search()` (line ~320)

```python
params = {
    ...
    # query_vector must be injected by caller; placeholder shows contract
    "query_vector": [],  ← hardcoded empty array
}
logger.warning(
    "federated_vector_search called without a real query vector; "
    "integrate with llm_service.get_embedding() before shipping"
)
```

The service itself acknowledges this in a warning log. The endpoint `POST /graphs/federate/vector-search` delegates to this method but never calls `llm_service.get_embedding()` to generate the query vector before passing it to the service. The Neo4j vector index query will receive an empty array and either error or return no results.

**Fix:** The `federated_vector_search` endpoint (or caller) must call `llm_service.get_embedding(query_text)` and pass the result as `query_vector` to `_execute_vector_search()`. The federation service's `_execute_vector_search()` parameter contract should be updated to accept `query_vector: List[float]` explicitly.

**Assign to:** Backend Developer Senior + AI Integration Specialist.

**Verdict: CONDITIONAL FAIL.** Entity search (`federated_query`) is completely blocked by BUG-1. Vector search is unimplemented. Once BUG-1 is fixed, the entity search path's architecture is sound.

---

### ORA-55 — Graph Versioning (`app/services/versioning_service.py`, `snapshot_service.py`, `rollback_service.py`)

**Implementation Coverage:**
- ✅ `GraphVersion` nodes with metadata: `version_id`, `version_number`, `captured_at`, `entity_count`, `relationship_count`, `label`, `description`, `is_auto`, `parent_version_id`
- ✅ Zero-copy snapshots — version is anchored by `captured_at` timestamp (no data duplication)
- ✅ Auto-increment `version_number` within graph scope (uses `coalesce(max(), 0) + 1`)
- ✅ Diff via transaction-time window queries for added/deleted entities and relationships
- ✅ Soft-invalidation rollback (`invalidated_at` semantics, not hard deletes)
- ✅ Async rollback path for large graphs (Celery task)
- ✅ Sync rollback path for small graphs
- ✅ Neo4j indexes: `(graph_id, captured_at)`, `(graph_id, version_number)`, relationship composite `(graph_id, transaction_time, invalidated_at)`
- ✅ Integration tests: 23 test functions in `test_versioning_api.py` covering CRUD, diff, rollback, cross-tenant isolation
- ✅ Multi-tenant isolation: cross-tenant snapshot access blocked (`test_cannot_access_snapshots_of_other_users_graph`)
- ✅ Endpoint: `diff_returns_404_for_missing_snapshot` tested

**Warning:**

#### WARN-3: `create_version` fallback creates orphaned `GraphVersion` nodes

**Location:** `app/services/versioning_service.py:create_version()` (lines ~136-168)

When the primary `CREATE` query fails to match the parent `Graph` node (line 103-119):
```cypher
MATCH (g:Graph {graph_id: $graph_id})
CREATE (v:GraphVersion {...})
CREATE (g)-[:HAS_VERSION]->(v)
```

A fallback query is executed that creates a `GraphVersion` node WITHOUT a `HAS_VERSION` relationship:
```cypher
CREATE (v:GraphVersion {...})
RETURN v
```

This creates orphaned `GraphVersion` nodes — versions not reachable via `MATCH (g:Graph)-[:HAS_VERSION]->(v)`. These orphaned nodes cannot be listed via `list_versions()` which queries `MATCH (v:GraphVersion {graph_id: $graph_id})` — but they CAN be found by that simpler query. So the listing actually works since it uses `graph_id` property, not the relationship.

The real issue is the orphaned graph relationship: if the codebase ever relies on `(g)-[:HAS_VERSION]->(v)` traversal (e.g., for graph deletion cascades), these orphaned versions would be missed.

**Verdict: CONDITIONAL PASS.** Versioning core logic is correct. Fallback path is a robustness concern, not a functional bug for current queries.

---

### ORA-56 — OpenTelemetry (`app/core/telemetry.py`)

**Implementation Coverage:**
- ✅ `setup_telemetry()` called once at FastAPI lifespan startup
- ✅ Opt-in via `OTEL_ENABLED=true` env var — disabled by default (no impact on non-OTEL deployments)
- ✅ Dual export: spans (OTLP gRPC or HTTP/protobuf) and metrics (Prometheus via OTLP)
- ✅ BatchSpanProcessor (async, buffered) — no blocking of request path
- ✅ `W3C TraceContext` propagation set globally
- ✅ FastAPI auto-instrumentation: `FastAPIInstrumentor.instrument_app()` — HTTP spans with endpoint, method, status_code
- ✅ Celery auto-instrumentation: `CeleryInstrumentor().instrument()` — task spans
- ✅ Custom metrics: 8 application-specific counters/histograms/up-down-counters (ingestion, chat, graph node/edge counts)
- ✅ `get_tracer(name)` and `get_meter(name)` exposed for use across services — no-op when disabled
- ✅ Graceful console fallback if OTLP exporter fails to connect
- ✅ `shutdown_telemetry()` flushes pending spans/metrics on app shutdown
- ✅ `current_trace_context()` for log correlation (trace_id + span_id injection)
- ✅ Integration tests in `test_telemetry_spans.py` verify span emission for Neo4j reads, writes, chat queries, and pipeline documents using `InMemorySpanExporter` (no Jaeger required)

**Warnings:**

#### WARN-4: Custom metrics created at startup but not actually recorded during operations

`_register_custom_metrics()` creates counters and histograms, but there are no corresponding `.add()` / `.record()` / `.observe()` calls found in the service layer (ingestion, chat, pipeline services). The metrics are defined but never incremented.

**Result:** Grafana dashboards would show all metrics at 0. This doesn't cause errors (OTel SDK handles this gracefully), but the feature is incomplete.

**Recommendation:** File a follow-up task for Backend Developer Senior to wire metric recording into: ingestion job completion (counter + duration histogram), entity extraction (entity count counter), and chat query handling (query counter + response time histogram).

#### WARN-5: No sampling configuration — all traces exported at 100% rate

No `TraceIdRatioBased` or `ParentBased` sampler is configured. At 100% sample rate under production load (50 concurrent users), trace volume may overwhelm the OTLP endpoint or Jaeger storage.

**Recommendation:** Configure a sampling ratio via `OTEL_TRACES_SAMPLER` env var before enabling in production. The `AlwaysOnSampler` (current default) is appropriate for development/staging only.

**Verdict: CONDITIONAL PASS.** Infrastructure is solid and opt-in. Metric recording is incomplete (data defined, never written). Both warnings are follow-up tasks, not blocking bugs.

---

### ORA-61 — Cypher Injection Fix (`app/components/multi_tenant_components.py`, `app/services/rebac_service.py`)

**Implementation Coverage:**
- ✅ All Cypher queries in `rebac_service.py` use parameterized variables (`$user_id`, `$graph_id`, etc.) — injection strings never appear in query text
- ✅ `test_cypher_query_uses_parameterized_variables` in `test_rebac.py` validates injection strings are not in query text:
  ```python
  user_id = "user'; DROP DATABASE neo4j; --"
  # asserts: injection string NOT in any query generated
  ```
- ✅ `multi_tenant_components.py`: graph_id injected as parameterized variable, never string-interpolated into Cypher
- ✅ `pipeline_service.py` and `retriever_factory.py`: reviewed — parameterized throughout
- ✅ `fulltext_index_service.py`: full-text search uses `apoc.index.fulltext.queryNodes` with parameterized query text

**Verdict: PASS.** Cypher injection is properly addressed via parameterized queries throughout the codebase.

---

### ORA-62 — JWT sub→UUID Migration (`auth-service/app/core/jwt_handler.py`)

**Implementation Coverage:**
- ✅ `_is_email()` detects legacy tokens where `sub` was set to email string
- ✅ `verify_access_token()` raises `HTTP 401 "Outdated token format — please re-authenticate"` for legacy tokens
- ✅ `create_access_token()` correctly documents that callers must pass `sub=str(user_id)` (UUID)
- ✅ `create_refresh_token()` similarly documented
- ✅ Service account tokens are short-lived (15 minutes) — separate from user tokens

**Warnings:**

#### WARN-5: No grace period for users with legacy tokens

Users who have unexpired tokens with email in `sub` (issued before the migration) will receive an immediate `HTTP 401` after deployment. There is no refresh-and-upgrade path — they must fully re-login. No client-side notification is surfaced.

**Impact:** All users logged in before the migration deployment will experience a forced logout. For a SaaS platform, this should be communicated as a breaking change.

**Recommendation:** DevOps SRE should coordinate a rotation window. The `outdated_token_exception` 401 response is distinct from `credentials_exception` — clients could detect this specific error and show a "Session updated — please log in" message. File as a UX polish task.

**Verdict: PASS.** Migration is correctly implemented and secure. The forced logout is a UX concern, not a security issue.

---

## Cross-Cutting Issues

#### CROSS-1 (P2-medium): Federation MATCH has no `namespace` filter on Graph nodes

`federation_service.py:_validate_and_filter()` uses:
```cypher
MATCH (g:Graph)
WHERE g.graph_id IN $graph_ids
```

The ReBAC service stores Graph nodes with `namespace: "__system__"` as a discriminator. If any non-ReBAC `:Graph` labeled nodes exist (e.g., from legacy migrations or test data), they could be matched, causing unexpected behavior. The MATCH should include `namespace: "__system__"` to be consistent with the ReBAC model.

---

## Test Coverage Assessment

| Feature | Unit Tests | Integration Tests | Gap |
|---|---|---|---|
| ORA-52 ReBAC | `tests/unit/test_rebac_service.py` | `tests/integration/test_rebac.py` (575 lines) | None critical |
| ORA-60 Incremental KG | `tests/unit/test_incremental_kg.py` (7 criteria) | **None** | Integration test gap — hash guard, stale chunks not E2E tested |
| ORA-54 Federation | Partial (entity union) | **None explicitly found** | **BUG-1 would be caught by a basic integration test** |
| ORA-55 Versioning | `tests/unit/test_versioning.py` | `tests/integration/test_versioning_api.py` (681 lines) | None critical |
| ORA-56 OTel | InMemorySpanExporter tests | `tests/integration/test_telemetry_spans.py` | Metric recording not tested |
| ORA-61 Injection | `test_cypher_query_uses_parameterized_variables` | Covered via multi-tenant tests | None critical |
| ORA-62 JWT | Auth service unit tests | Partial | No E2E test for legacy token rejection |

**Root cause of BUG-1 going undetected:** No integration test exercises `POST /graphs/federate/query` against a real Neo4j instance with actual Graph nodes. The federation service is tested at the unit level but not end-to-end.

---

## Bugs Filed

| Priority | Feature | Title | Assign To |
|---|---|---|---|
| P1-critical | ORA-54 | Federation `_validate_and_filter` reads `g.user_id` (None) — should be `g.owner_user_id` — all federated queries return 403 | Backend Developer Senior |
| P2-medium | ORA-54 | Federation vector search passes empty `query_vector: []` — feature non-functional | Backend Developer Senior + AI Integration Specialist |
| P2-medium | ORA-56 | OTel custom metrics created but never recorded — dashboards always show 0 | Backend Developer Senior |

---

## Recommendations

1. **Immediate (P1):** Fix `g.user_id` → `g.owner_user_id` in `federation_service.py:_validate_and_filter()`. One-line fix, but it unblocks the entire federation feature.

2. **Short-term (P2):** Wire `llm_service.get_embedding(query_text)` into the federation vector search endpoint before the vector search feature is released. The service layer correctly accepts `query_vector` — it just needs the LLM call injected by the caller.

3. **Short-term (P2):** Add integration tests for federation (`POST /graphs/federate/query` with a real Neo4j graph with `federatable=true`). This would have caught BUG-1 immediately.

4. **Short-term (P2):** Add integration tests for incremental KG re-ingestion against real Neo4j. Unit tests cover logic, but the Cypher delta queries need E2E validation.

5. **Medium-term (P3):** Wire OTel metric recording into ingestion, chat, and pipeline services. The metric definitions are done — only the `.add()` / `.record()` calls are missing.

6. **Medium-term (P3):** Configure OTel sampling ratio for production (`TraceIdRatioBased` or `ParentBased` with 10–20% ratio). Do not deploy with `AlwaysOnSampler` at production scale.

7. **Operational (P3):** Document the Phase A→B transition contract in the ReBAC architecture doc. Any future migration that bootstraps Phase B for existing graphs MUST migrate CAN_ACCESS → HAS_ROLE simultaneously.

---

## QA Gate Verdict

| Gate | Status |
|---|---|
| Integration tests pass | ✅ (ReBAC, Versioning) / ❌ (Federation has P1 bug, no integration test) |
| Multi-tenant isolation | ✅ Verified for all features (cross-tenant versioning, ReBAC scoping, federation fail-closed) |
| No regression in existing functionality | ✅ |
| Security regression tests | ✅ Cypher injection confirmed blocked; JWT migration correctly enforced |
| Federation functional | ❌ BUG-1 blocks all user-path federation queries |

**Overall:** ORA-52 (ReBAC), ORA-56 (OTel), ORA-60 (Incremental KG), ORA-55 (Versioning), ORA-61 (Cypher injection), ORA-62 (JWT migration) — PASS with warnings. ORA-54 (Federation) — FAIL due to P1 property name mismatch. Phase 3 cannot be considered fully complete until BUG-1 is resolved and a basic federation integration test is added.
