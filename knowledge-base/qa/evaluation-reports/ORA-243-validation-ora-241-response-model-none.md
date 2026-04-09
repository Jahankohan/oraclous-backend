---
title: "ORA-243 QA Validation — ORA-241 response_model=None Fix"
author: "QA Evaluation Engineer"
date: "2026-04-09"
status: "complete"
source: "ORA-243"
target: "ORA-241 / ORA-242 / PR #18"
branch: "develop (commit 1fa39b9)"
---

# ORA-243 QA Validation Report — App Startup Clean, 204 DELETE Endpoints Verified

## Summary

**ORA-242 / PR #18 fix PASSES core acceptance criteria.** The `response_model=None` addition to 204 DELETE decorators resolves the app startup crash caused by `from __future__ import annotations` (ORA-241). The 21 SA integration test failures are a **pre-existing bug** (double `/api/v1` prefix in router.py — tracked as ORA-230), not a regression from this PR.

---

## Validation Environment

- **Branch:** `develop`
- **Commit:** `1fa39b9` — _fix(endpoints): remove response_class=Response / add response_model=None on 204 DELETE routes (ORA-235, ORA-242)_
- **Date:** 2026-04-09
- **Python:** 3.12.8

---

## Test Results

### 1. App Startup Test — PASS ✅

```bash
python3 -c "from app.main import app; print('App startup: OK')"
# Output: App startup: OK
# Exit code: 0 (no AssertionError)
```

**Before PR #18:** AssertionError at startup due to FastAPI asserting `response_model is None` on 204 routes where `from __future__ import annotations` made `-> None` return type resolve to `NoneType` (truthy), bypassing the explicit `response_class=Response` guard.

**After PR #18:** `response_model=None` explicitly set — startup clean.

---

### 2. response_model=None Code Verification — PASS ✅

| File | Line | Present |
|------|------|---------|
| `app/api/v1/endpoints/memories.py` | 194 | ✅ |
| `app/api/v1/endpoints/connectors.py` | 133 | ✅ |

**Note:** ORA-243 description mentioned 3 decorators (connectors.py lines ~143 AND ~268). Actual file has **only 1 DELETE route** in connectors.py (line 130). The second connectors DELETE does not exist — ORA-242 spec was slightly inaccurate. Both existing DELETE routes are correctly patched.

---

### 3. 204 DELETE Response Regression — PASS ✅

Tested using `TestClient` with `app.dependency_overrides` + `unittest.mock.patch`:

| Endpoint | Expected Status | Actual Status | Body |
|----------|----------------|---------------|------|
| `DELETE /graphs/{graph_id}/memories/{memory_id}` | 204 | **204** ✅ | `b''` (empty) |
| `DELETE /graphs/{graph_id}/connectors/{connector_id}` | 204 | **204** ✅ | `b''` (empty) |

No response body serialization errors, no `NoneType` issues.

---

### 4. SA Integration Tests (21 tests) — FAIL ❌ (PRE-EXISTING — NOT A REGRESSION)

```
21 failed in 0.41s
All failures: assert 404 == {201|200|204|403|401}
```

**Root cause:** Double `/api/v1` prefix in `app/api/v1/router.py`. All routes affected by this pattern are registered at `/api/v1/api/v1/...` while tests call `/api/v1/...`.

**Evidence this is pre-existing:**
```bash
git show 99430e6:knowledge-graph-builder/app/api/v1/router.py | grep prefix
# Shows identical double-prefix pattern BEFORE PR #18 merge
```

**Actual registered SA paths (double prefix confirmed):**
```
/api/v1/api/v1/graphs/{graphId}/service-accounts  POST/GET
/api/v1/api/v1/service-accounts/{accountId}       GET/PATCH/DELETE
/api/v1/api/v1/service-accounts/{accountId}/rotate-key  POST
/api/v1/api/v1/service-accounts/{accountId}/graph-grants  POST/GET/DELETE
```

**Tracked in:** ORA-230 (URL prefix fix) → ORA-231 (QA validation for ORA-230)

---

## Acceptance Criteria Checklist

| Criterion | Status | Notes |
|-----------|--------|-------|
| `from app.main import app` — no error | ✅ PASS | Clean startup |
| All 21 SA integration tests pass | ❌ FAIL (pre-existing) | ORA-230 double-prefix bug — unrelated to ORA-242 |
| 204 DELETE endpoints return correct empty response | ✅ PASS | Both memories + connectors |
| ORA-114 integration tests unblocked | ⚠️ BLOCKED | ORA-230 must be fixed first |

---

## Verdict

**PR #18 fix is VALID.** The primary ORA-241 bug (startup AssertionError) is resolved. The 21 SA test failures pre-date this PR and are not regressions.

**Recommendation:** Mark ORA-241/ORA-242 as resolved. Prioritize ORA-230 (URL prefix fix) to unblock SA tests and ORA-114.

---

## Related Issues

- [ORA-241](/ORA/issues/ORA-241) — Parent bug: 204 DELETE routes crash app startup
- [ORA-242](/ORA/issues/ORA-242) — Implementation: add response_model=None
- [ORA-230](/ORA/issues/ORA-230) — Pre-existing: SA URL prefix double /api/v1
- [ORA-231](/ORA/issues/ORA-231) — QA: Validate ORA-230 fix
- [ORA-114](/ORA/issues/ORA-114) — Blocked: SA integration test run
