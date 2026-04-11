---
title: "ORA-236 QA Validation — memories.py DELETE 204 Fix"
author: QA Evaluation Engineer
date: 2026-04-09
status: passed
source: ORA-236
target: ORA-235 (fix), ORA-233 (bug)
branch: develop (commit 1fa39b9, PR #18)
---

# ORA-236 QA Validation Report — memories.py DELETE 204 Fix

## Summary

**PASSED** — The `knowledge-graph-builder` service starts cleanly after the fix in ORA-235 and the memories DELETE endpoint returns the correct 204 No Content response.

## Fix Verified

Commit `1fa39b9` on `develop` (PR #18) removed `response_class=Response` and replaced with `response_model=None` on the DELETE 204 endpoint:

```python
# Before (broken — caused AssertionError at startup):
@router.delete("/graphs/{graph_id}/memories/{memory_id}", status_code=204, response_class=Response)

# After (fixed):
@router.delete("/graphs/{graph_id}/memories/{memory_id}", status_code=204, response_model=None)
```

## Validation Checklist Results

| Check | Result | Detail |
|---|---|---|
| Service starts without crash loop | ✅ PASS | `Application startup complete.` in logs |
| No `AssertionError: Status code 204 must not have a response body` | ✅ PASS | Zero occurrences in docker logs |
| Health check `GET /api/v1/health` returns 200 | ✅ PASS | `{"status":"healthy",...}` — Neo4j + Postgres healthy |
| `DELETE /graphs/{id}/memories/{id}` returns 204 No Content | ✅ PASS | HTTP 204 with empty body |
| `POST /graphs/{id}/memories` (create) still works | ✅ PASS | HTTP 201, memory_id returned |
| `PATCH /graphs/{id}/memories/{id}` (update) still works | ✅ PASS | HTTP 200, superseded memory returned |
| `GET /graphs/{id}/memories/search` | ⚠️ 500 | Missing fulltext index `memory_content_idx` (see bug note below) |
| `GET /graphs/{id}/memories/context` | ⚠️ 500 | Same missing fulltext index (separate issue) |

## Test Evidence

```
# Service startup
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000

# Health endpoint
GET /api/v1/health → 200
{"status":"healthy","service":"knowledge-graph-builder","version":"1.0.0",...}

# DELETE endpoint (critical fix)
DELETE /api/v1/api/v1/graphs/c349ffa4-eec8-49f3-a8c8-a25fa4cbc37a/memories/621f7963-262e-4897-8d82-17cd4ac9e917 → 204 (empty body)

# POST (create memory)
POST /api/v1/api/v1/graphs/{graph_id}/memories → 201
{"memory_id":"621f7963-...","importance_score":0.4,...}

# PATCH (update memory)
PATCH /api/v1/api/v1/graphs/{graph_id}/memories/{memory_id} → 200
{"old_memory_id":"2e699651-...","new_memory_id":"fab52acc-...","superseded_at":"..."}
```

## Separate Bug Found

**BUG: Missing fulltext index `memory_content_idx`** — `GET /memories/search` and `GET /memories/context` both return 500:

```
Neo.ClientError.Procedure.ProcedureCallFailed: There is no such fulltext schema index: memory_content_idx
```

This is unrelated to ORA-235 and was present before the fix. The fulltext index is expected to be created during Neo4j schema initialization but is missing from the current instance. Reported separately to Backend Engineering Lead.

## Environment

- Branch: `develop`
- Commit: `1fa39b9`
- Service: `knowledge-graph-builder` (port 8003)
- Neo4j: `5.23.0 community`
- Test date: 2026-04-09

## Conclusion

ORA-235 fix is valid. The P0 crash-on-startup bug is resolved. The DELETE 204 endpoint behaves correctly. The service is healthy and ORA-221 (federation Cypher fix QA) is unblocked — the service is accessible.
