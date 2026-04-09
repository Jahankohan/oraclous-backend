---
title: "QA Smoke Test Report — ORA-108: Federation 403 Fix (ORA-102)"
author: "QA Evaluation Engineer"
date: "2026-04-09"
status: "complete"
source: "ORA-102 — g.owner_user_id fix in federation_service.py"
target: "POST /api/v1/graphs/federate/query"
branch_tested: "develop (commit ed08ea6, post 67bd406 fix)"
---

# QA Smoke Test Report — ORA-108

**Task:** [ORA-108](/ORA/issues/ORA-108) — Smoke test for ORA-102 federation 403 fix
**Parent bug:** [ORA-102](/ORA/issues/ORA-102)
**Test date:** 2026-04-09
**Branch tested:** `develop` (includes commit `67bd406` — `fix(federation): read g.owner_user_id instead of g.user_id`)
**Environment:** Docker Compose stack (auth-service:8000, knowledge-graph-builder:8003, neo4j:7687)

---

## Unit Test Verification — PASSED (2/2)

Both named regression tests from the task description pass:

| Test | Result |
|---|---|
| `test_validate_uses_owner_user_id_not_user_id` | ✅ PASSED |
| `test_validate_matches_system_namespace` | ✅ PASSED |

Run inside container: `pytest tests/unit/test_federation_service.py::test_validate_uses_owner_user_id_not_user_id tests/unit/test_federation_service.py::test_validate_matches_system_namespace` — **2 passed in 0.02s**

---

## ORA-102 Fix Verification — CONFIRMED

**What the fix changed:**
- `MATCH (g:Graph)` → `MATCH (g:Graph {namespace: "__system__"})`
- `g.user_id AS user_id` → `g.owner_user_id AS user_id`

**Why:** `g.user_id` is always `None` on ReBAC-managed Graph nodes (rebac_service stores ownership in `g.owner_user_id`). The stale property caused every user-path federated query to fail the ownership check and raise 403.

**Evidence of fix working:** Same-user federatable query no longer receives 403. It now passes validation and proceeds to the entity search phase (where a separate secondary bug produces a 500 — see Bug Filing below). The key behavioral change is confirmed: **the false 403 is gone**.

---

## Live API Smoke Test Results

Test graphs created directly in Neo4j (graph creation API blocked by secondary infrastructure bug — see below):

| Graph ID | Owner | federatable | namespace |
|---|---|---|---|
| `qa-smoke-graph-a` | user1 | `true` | `__system__` |
| `qa-smoke-graph-b` | user1 | `true` | `__system__` |
| `qa-smoke-graph-c` | user1 | `false` | `__system__` |
| `qa-smoke-graph-d-user2` | user2 | `true` | `__system__` |

### Step 2-3: Same-user federatable query
- **Request:** `POST /api/v1/api/v1/graphs/federate/query` with `graph_ids: [qa-smoke-graph-a, qa-smoke-graph-b]`
- **Expected:** 200 OK
- **Actual:** 500 Internal Server Error
- **Root cause:** Validation passed (ORA-102 fix confirmed), but `_execute_entity_union` uses unsupported Cypher `CALL {} UNION ALL CALL {}` syntax — Neo4j 5.23 error: `"Query cannot conclude with CALL"`. **This is a separate bug (ORA-212), not ORA-102.**
- **QA interpretation:** The ORA-102 fix is confirmed working (no false 403). The 500 is unrelated to the auth fix.

### Step 4: Entities from both graphs in results
- **Status:** Blocked by Step 2-3 Cypher bug (ORA-212)

### Step 5: Cross-tenant isolation — ✅ PASSED
- **Request:** user1 queries `[qa-smoke-graph-a, qa-smoke-graph-d-user2]`
- **Expected:** 403 Forbidden
- **Actual:** `403` — `"Access denied — no accessible graphs in federation request"`
- **Result:** ✅ Cross-tenant block preserved

### Step 6: Non-federatable graph rejection — ✅ PASSED
- **Request:** user1 queries `[qa-smoke-graph-a, qa-smoke-graph-c]` (graph-c has `federatable=false`)
- **Expected:** 400 Bad Request
- **Actual:** `400` — `"One or more requested graphs are not enabled for federation"`
- **Result:** ✅ Non-federatable rejection preserved

---

## QA Gate Summary

| Gate | Result | Notes |
|---|---|---|
| Named regression tests (2/2) | ✅ PASS | `test_validate_uses_owner_user_id_not_user_id` + `test_validate_matches_system_namespace` |
| False 403 eliminated | ✅ PASS | Same-user federatable query passes validation |
| Cross-tenant block (Step 5) | ✅ PASS | Returns 403 as required |
| Non-federatable rejection (Step 6) | ✅ PASS | Returns 400 as required |
| Full 200 OK with entities (Steps 3-4) | ⚠️ BLOCKED | Blocked by ORA-212 (Cypher UNION ALL syntax bug, separate from ORA-102) |

**ORA-102 fix verdict: APPROVED** — the authentication/authorization fix is correct and all regression guards pass. The entity retrieval 500 is a separate pre-existing issue.

---

## Bugs Filed

### ORA-212 (P1) — Federation entity search: invalid Cypher UNION ALL syntax
- `_execute_entity_union` generates `CALL {} UNION ALL CALL {}` which is not valid in Neo4j 5.23
- Error: `Query cannot conclude with CALL (must be a RETURN clause...)`
- Blocks full federation entity retrieval but does NOT affect auth validation
- **Assigned to:** Backend Engineering Lead for triage

### ORA-XXX (P2) — `sync_driver` not initialized at startup
- `main.py` lifespan only calls `connect_async()` — `connect_sync()` is never called
- All endpoints using `sync_driver` (graph creation, etc.) return 503 on fresh container start
- `POST /api/v1/graphs` returns `503 "Neo4j connection not available"` until a Celery job triggers `connect_sync()`
- **Assigned to:** Backend Engineering Lead for triage

### Build hygiene observation
- The oraclous-data-studio repo was on `fix/phase-4/sr-backend/ora-106-sa-integration-tests` when Docker build ran
- The ORA-102 fix only exists on `develop`
- First build produced an image WITHOUT the fix — required branch switch to `develop` before rebuild
- **Recommendation:** CI/CD should always build from `develop`; add branch guard to build scripts

---

## Infrastructure Notes

- Rebuilt knowledge-graph-builder image from `develop` branch (commit `ed08ea6`)
- auth-service was already rebuilt (27 min prior) with passlib/bcrypt fix (ORA-181)
- All core services healthy at test time: neo4j ✅, postgres ✅, redis ✅, auth-service ✅
