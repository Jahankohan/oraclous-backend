# Deployment Validation Report — Full-Stack Docker Compose

**Date:** 2026-04-08
**Task:** ORA-96 — Full-Stack Deployment Validation (All Phases)
**Validated by:** DevOps SRE Specialist
**Branch:** `feature/phase-4/sr-backend/ora-105-multimodal-integration-tests` (current develop state)

---

## Validation Result: ❌ FAILED — 3 Critical/High Bugs Found

3 services are non-functional. All 3 bugs have been filed as sub-tasks of ORA-96 and assigned to Backend Lead / Backend Senior Developer for immediate resolution.

---

## Infrastructure Services — All Healthy ✅

| Service | Port | Status | Notes |
|---------|------|--------|-------|
| Neo4j | 7474, 7687 | ✅ Healthy | APOC + GDS plugins loaded |
| PostgreSQL | 5432 | ✅ Healthy | `pg_isready` passing |
| Redis | 6379 | ✅ Healthy | `PONG` |
| Jaeger | 16686 | ✅ Healthy | UI accessible, OTLP receivers on 4317/4318 |

---

## Application Services

| Service | Port | Status | Health Endpoint | Notes |
|---------|------|--------|-----------------|-------|
| oraclous-core-service | 8001 | ✅ Up | `/api/v1/health` → 200 | `{"status":"healthy","version":"1.0.0"}` |
| credential-broker-service | 8002 | ✅ Up | `/health` → 200 | Note: endpoint is `/health`, not `/api/v1/health` |
| auth-service | 8000 | ❌ BROKEN | N/A (no healthcheck) | Silent failure — see ORA-163 |
| knowledge-graph-builder | 8003 | ❌ CRASH LOOP | `/api/v1/health` unreachable | Import error at startup — see ORA-162 |
| knowledge-graph-worker | — | ❌ UNHEALTHY | N/A (Celery, no HTTP) | Dockerfile healthcheck misconfigured — see ORA-164 |
| knowledge-graph-mcp | 8004 | ❌ NOT STARTED | — | Blocked by `knowledge-graph-builder` being unhealthy |

---

## Bug Reports Filed

### ORA-162 — CRITICAL — knowledge-graph-builder crash loop
**Assigned to:** Backend Developer Senior
**File:** `knowledge-graph-builder/app/api/v1/endpoints/connectors.py:122`

```
AssertionError: Status code 204 must not have a response body
```

The `delete_connector` DELETE endpoint uses `status_code=HTTP_204_NO_CONTENT` without `response_class=Response`. In current FastAPI version, this triggers a startup assertion.

**Fix:** Add `response_class=Response` to the `@router.delete(...)` decorator at line 122.

---

### ORA-163 — CRITICAL — auth-service silently broken (ORA-135 regression)
**Assigned to:** Backend Lead Developer
**File:** `auth-service/app/main.py:4`

```
ModuleNotFoundError: No module named 'slowapi'
```

ORA-135 introduced `from slowapi.errors import RateLimitExceeded` but `slowapi` was not added to `requirements.txt`. The container appears `Up` (no healthcheck, uvicorn dev hot-reload keeps process alive) but is completely non-functional.

**Fix:** Add `slowapi>=0.1.9` to `auth-service/requirements.txt` and rebuild.
**DevOps follow-up:** Add healthcheck to auth-service in `docker-compose.yml` after fix.

---

### ORA-164 — HIGH — knowledge-graph-worker Dockerfile healthcheck misconfiguration
**Assigned to:** Backend Lead Developer

```
curl: (7) Failed to connect to localhost port 8000
FailingStreak: 26
```

The worker Dockerfile healthcheck uses `curl localhost:8000/api/v1/health` — but Celery workers do not expose an HTTP server. The Celery process itself is running correctly.

**Fix:** Replace Dockerfile healthcheck with `celery inspect ping` or remove it.

---

## Smoke Test Results

| Test | Result | Notes |
|------|--------|-------|
| Auth endpoint reachable | ❌ FAIL | Service non-functional (slowapi missing) |
| Graph CRUD API | ❌ FAIL | knowledge-graph-builder crash loop |
| Chat endpoint | ❌ FAIL | Depends on knowledge-graph-builder |
| Ingestion pipeline | ❌ FAIL | knowledge-graph-builder down |
| MCP endpoint | ❌ FAIL | knowledge-graph-mcp not started |
| Multi-tenant isolation | ❌ NOT TESTED | Blocked by builder crash |
| OTel traces (Jaeger) | ✅ INFRA READY | Jaeger healthy; no traces generated due to service failures |
| oraclous-core-service health | ✅ PASS | 200 OK |
| credential-broker health | ✅ PASS | 200 OK |
| Neo4j connectivity | ✅ PASS | Bolt + HTTP healthy |

---

## Required Actions Before Re-Validation

1. **ORA-162** (critical) — Fix `connectors.py` 204 assertion → Backend Developer Senior
2. **ORA-163** (critical) — Add `slowapi` to auth-service requirements → Backend Lead Developer
3. **ORA-164** (high) — Fix Celery Dockerfile healthcheck → Backend Lead Developer
4. Once above are merged to develop, rebuild images and re-run this validation

---

## DevOps Improvements Identified

These are tracked separately and will be implemented after the above bugs are resolved:

1. **Add auth-service healthcheck** to `docker-compose.yml` — prevents silent "Up but broken" failures
2. **Consider `required: false` env_file syntax** (Docker Compose v2.24+) — avoids base `.env` dependency issue on staging servers (noted in ORA-149 review)
