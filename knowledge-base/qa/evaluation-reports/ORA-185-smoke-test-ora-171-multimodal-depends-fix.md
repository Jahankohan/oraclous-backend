---
title: "ORA-185 — Smoke Test: ORA-171 Multimodal verify_graph_access Bug Fix"
author: QA Evaluation Engineer
date: 2026-04-08
status: passed
source: fix/phase-1/lead-backend/ora-171-multimodal-depends-fix
target: ORA-171 — multimodal endpoint calls verify_graph_access directly
---

# ORA-185 — Smoke Test Report: ORA-171 Multimodal `verify_graph_access` Bug Fix

## Summary

**Result: PASSED** ✅

All acceptance criteria met. The fix converts direct `verify_graph_access()` calls to `Depends(verify_graph_write_access)` in both multimodal endpoints, enabling FastAPI's dependency injection to function correctly in tests.

## Branch Validated

`fix/phase-1/lead-backend/ora-171-multimodal-depends-fix`
Commit: `abc9bf6 fix(multimodal): convert direct verify_graph_access calls to Depends`

## Checklist

| # | Check | Result |
|---|---|---|
| 1 | Integration tests: 11/11 pass | ✅ PASS |
| 2 | Was 2/11 before fix, now 11/11 | ✅ CONFIRMED |
| 3 | No unit test regressions introduced | ✅ NO REGRESSION |
| 4 | Cross-tenant isolation: 403 returned correctly | ✅ PASS |
| 5 | CI green on branch | ⚠️ CI INFRASTRUCTURE BLOCKED (see below) |

## Integration Test Results

```
tests/integration/test_multimodal_api.py::test_ingest_document_pdf_success        PASSED
tests/integration/test_multimodal_api.py::test_ingest_image_png_success           PASSED
tests/integration/test_multimodal_api.py::test_ingest_document_too_large          PASSED
tests/integration/test_multimodal_api.py::test_ingest_image_too_large             PASSED
tests/integration/test_multimodal_api.py::test_ingest_document_invalid_mime       PASSED
tests/integration/test_multimodal_api.py::test_ingest_image_invalid_mime          PASSED
tests/integration/test_multimodal_api.py::test_ingest_document_missing_upload_dir PASSED
tests/integration/test_multimodal_api.py::test_ingest_document_neo4j_unavailable  PASSED
tests/integration/test_multimodal_api.py::test_ingest_image_neo4j_unavailable     PASSED
tests/integration/test_multimodal_api.py::test_ingest_document_cross_tenant_denied PASSED
tests/integration/test_multimodal_api.py::test_ingest_image_cross_tenant_denied   PASSED

11 passed in 0.79s
```

Command: `cd knowledge-graph-builder && PYTHONPATH=. pytest tests/integration/test_multimodal_api.py -m integration -v`

## Unit Test Regression Check

- **Branch (ORA-171 fix):** 673 passed, 33 failed
- **`develop` baseline:** 673 passed, 33 failed
- **Delta: 0 regressions**

The 33 pre-existing failures are tracked separately and are NOT caused by this fix. Known failing suites:
- `test_ora138_smoke.py` — temporal filter logic (pre-existing, ORA-138 related)
- `test_service_accounts.py` — coroutine attribute errors (pre-existing)
- `test_rate_limiting.py` — IP isolation test (pre-existing)
- `test_pipeline_service.py` — approx() usage issue (pre-existing)

## Cross-Tenant Isolation Verification

`test_ingest_document_cross_tenant_denied` and `test_ingest_image_cross_tenant_denied` both return **HTTP 403** as expected when a user attempts to ingest into a graph owned by another user. The `Depends(verify_graph_write_access)` injection correctly routes through the DI override in tests.

## Root Cause — What Was Fixed

**Before (broken):**
```python
# multimodal.py — direct function call, bypasses DI overrides
await verify_graph_access(str(graph_id), "write", user_id)
```

**After (fixed):**
```python
# multimodal.py — injected via Depends, DI overrides work correctly
async def ingest_document(
    ...
    _access: str = Depends(verify_graph_write_access),
    ...
):
```

## CI Status

GitHub Actions CI runs are failing across **all** branches due to 192 ruff lint errors (tracked in `fix/ci/lead-backend/ruff-lint-errors`). This is a CI infrastructure issue unrelated to ORA-171. Local test execution confirms the fix is correct.

## Verdict

**QA GATE: PASSED** — the ORA-171 fix is correct and may be merged to `develop`. The CI infrastructure blocker (ruff lint) should be resolved independently.
