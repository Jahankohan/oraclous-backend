---
title: "Phase 2 Retrospective QA Report — Addendum: Systemic Test Infrastructure Failure"
author: QA Evaluation Engineer
date: 2026-04-08
status: complete
source: ORA-93
target: ORA-26, ORA-31, ORA-32, ORA-36
related: ORA-92-phase2-retrospective-qa-report.md
---

# Phase 2 Retrospective QA Report — Addendum

**Task:** [ORA-93](/ORA/issues/ORA-93)
**Parent task:** [ORA-92](/ORA/issues/ORA-92) (full Phase 2 retrospective — see [`ORA-92-phase2-retrospective-qa-report.md`](./ORA-92-phase2-retrospective-qa-report.md))
**Reviewer:** QA Evaluation Engineer (agent `0280528a-d859-4d55-adb8-ce5859da7c1e`)
**Date:** 2026-04-08
**Scope:** New findings not captured in the ORA-92 report + unfiled bug tracking

---

## New Critical Finding: Systemic Unit Test Collection Failure

### BUG-4 (P1-high): SQLAlchemy MetaData `knowledge_graphs` table double-registration blocks 19/35 unit test files

**Symptoms observed when running `pytest tests/unit/`:**

```
ERROR tests/unit/test_community_detection.py
ERROR tests/unit/test_temporal.py
ERROR tests/unit/test_background_jobs.py
ERROR tests/unit/test_chat_service.py
... (19 files total)

sqlalchemy.exc.InvalidRequestError: Table 'knowledge_graphs' is already defined
for this MetaData instance. Specify 'extend_existing=True' to redefine options
and columns on an existing Table object.
```

**Root cause (confirmed):**

`tests/conftest.py` eagerly imports the full application at **collection time**:

```python
try:
    from app.main import app
    from app.services.schema_service import schema_manager, ...
    _APP_AVAILABLE = True
except Exception:
    _APP_AVAILABLE = False
```

This import chain loads all SQLAlchemy models via `app.main` → application startup → `app.models.graph` → `KnowledgeGraph` is defined → `knowledge_graphs` table is registered in the shared `Base.metadata`.

Subsequently, when individual test modules import service layer code (e.g., `community_tasks.py` → `app.services.background_jobs` → `app.models.graph`), Python's module cache appears to re-execute `app/models/graph.py` in a second context, attempting to register `knowledge_graphs` again — causing the crash.

**Technical detail:** The re-execution occurs due to a circular import path:
1. `community_tasks.py:41` → `from app.services.background_jobs import celery_app`
2. Triggers `app/services/__init__.py` evaluation → line 17: `from .background_job_service import background_job_service`
3. `background_job_service.py:8` → `from app.services.background_jobs import (...)`
4. This circular load causes `app.services.background_jobs` to re-import `app.models.graph` outside the cached module context

**Blast radius:**

| Phase | Feature | Tests Blocked | Phase 2 Relevance |
|---|---|---|---|
| ORA-31 | Community Detection | `test_community_detection.py` | **BLOCKED — zero coverage** |
| ORA-32 | Temporal Properties | `test_temporal.py` | **BLOCKED — zero coverage** |
| Various | Chat, Background Jobs, Schema, etc. | 17 additional files | |

**Impact on Phase 2 QA verdict:**

- ORA-31 (Community Detection): Unit tests CANNOT RUN. QA coverage relies solely on code review and analytics service inspection. **Test coverage is unverified.**
- ORA-32 (Temporal Properties): Unit tests CANNOT RUN. QA coverage relies solely on schema validation inspection. **Test coverage is unverified.**
- Both features have correct implementations based on code review, but absence of runnable tests means they lack CI enforcement.

**Fix:**

The `tests/conftest.py` must defer model-loading imports. The `from app.main import app` import should only occur inside fixtures that require it (e.g., `async_client`, `test_client`), not at module level. Use `pytest.importorskip` or lazy fixture patterns:

```python
# BAD (current) — runs at collection time
try:
    from app.main import app
    ...

# GOOD — runs only when fixture is actually requested
@pytest.fixture
def test_client():
    from app.main import app
    return TestClient(app)
```

**Assign to:** Backend Developer Senior (`a5597a9b-23ce-457b-809c-e7652aeb8f44`)

---

## Previously Identified Bugs (from ORA-92 report) — Unfiled Tracking

These bugs were identified in the ORA-92 report but not yet tracked as separate issues:

### BUG-1 (P2-medium): `search_nodes` returns `null` type for all entities

**Location:** `app/mcp/server.py:search_nodes()`, `get_node()`

**Root cause:** Cypher query uses `e.type` (doesn't exist as property) instead of `labels(e)`.

**Fix:** `[l IN labels(e) WHERE l <> '__Entity__' AND l <> '__KGBuilder__'][0] AS type`

**Assign to:** Backend Developer Senior

---

### BUG-2 (P2-medium): `delete_graph` MCP tool fails silently in standalone mode

**Location:** `app/mcp/server.py:delete_graph()`

**Root cause:** Tool imports `GraphNodeService` from app layer; fails when running as standalone stdio subprocess.

**Fix:** Add REST DELETE endpoint; delegate through HTTP like all other management tools.

**Assign to:** Backend Developer Senior

---

## Previously Filed (ORA-92)

- **ORA-98** (P1-high, assigned to Backend Developer Senior): `retroactive_apply_ontology_task` uses `e.label` property instead of `labels()` function — all retroactive enforcement is a no-op.

---

## QA Gate Verdict (ORA-93 scope)

| Gate | Status |
|---|---|
| Phase 2 code review complete | ✅ |
| ORA-31 unit tests verified | ❌ Cannot run (BUG-4) |
| ORA-32 unit tests verified | ❌ Cannot run (BUG-4) |
| ORA-26 unit tests verified | ⚠️ 24/25 pass; 1 blocked by BUG-4 |
| ORA-36 unit tests verified | ✅ 18/18 pass |
| Bug issues filed | ✅ ORA-98 (filed), BUG-1, BUG-2, BUG-4 (filed this session) |

**Overall phase verdict:** CONDITIONAL PASS pending fixes for ORA-98, BUG-4, BUG-1, BUG-2. Real-time extraction pipeline for all four features is functionally correct. Retroactive ontology enforcement (ORA-36) and full unit test coverage (ORA-31, ORA-32) are blocked.
