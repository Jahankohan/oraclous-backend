---
title: "ORA-250 QA Validation — knowledge-graph-mcp SSE Healthcheck Fix (ORA-249)"
author: QA Evaluation Engineer
date: 2026-04-09
status: pass-with-note
source: ORA-250
target: ORA-249 (fix/phase1/devops-sre/ora-248-mcp-healthcheck-sse)
---

# QA Validation Report — ORA-250

## Summary

**Result: PASS (code-verified + live command test)** — fix is correct and will resolve the unhealthy status once PR #19 is merged and the container is recreated. The running stack still uses the old config (pre-merge), so the live container currently shows `unhealthy`.

---

## Context

| Item | Value |
|---|---|
| Fix branch | `fix/phase1/devops-sre/ora-248-mcp-healthcheck-sse` |
| PR | https://github.com/Jahankohan/oraclous-backend/pull/19 (→ `develop`) |
| Commit | `596b19f` |
| Parent bug | ORA-248 — healthcheck curl times out on SSE stream |
| Prior port fix | ORA-228 — wrong port 8000 vs 8004 |
| Validated on | 2026-04-09, Docker 28.2.2, Docker Compose v2.36.2 |

---

## Validation Steps Executed

### 1. Code Review — PASS

Diff between `origin/develop` and fix branch on `docker-compose.yml`:

**Before (broken):**
```yaml
# No healthcheck block in docker-compose.yml for knowledge-graph-mcp
# Container falls back to Dockerfile HEALTHCHECK targeting :8000
# Previously ORA-228 added: test: ["CMD", "curl", "-f", "http://localhost:8004/sse"]
```

**After (fixed):**
```yaml
healthcheck:
  test: ["CMD-SHELL", "bash -c 'curl -sf --max-time 3 http://localhost:8004/sse; code=$?; [ $code -eq 0 ] || [ $code -eq 28 ]'"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 15s
```

Fix analysis:
- `CMD-SHELL` form required for bash scripting ✅
- `--max-time 3` forces curl to exit after 3s (before 10s Docker timeout kills it) ✅
- Exit code 28 = CURLE_OPERATION_TIMEDOUT (HTTP 200 received, stream held open) — accepted as success ✅
- `timeout: 10s` > `--max-time 3` — curl self-exits before Docker kills it ✅
- Exit code 7 (connection refused) is NOT accepted — genuine failures still fail ✅

### 2. curl Availability Check — PASS

Dockerfile explicitly installs curl:
```dockerfile
RUN apt-get update && apt-get install -y curl gcc libigraph-dev
```
`knowledge-graph-mcp` uses the same Dockerfile as `knowledge-graph-builder` — curl is available.

### 3. SSE Endpoint Live Test — PASS

```
$ curl -s --max-time 3 http://localhost:8004/sse
event: endpoint
data: /messages/?session_id=f962d31c07644ea78661e182d6d098e5
EXIT:28
```
SSE stream at `/sse` returns HTTP 200 + event data immediately, holds connection open as expected. Exit code 28 is correct.

### 4. New Healthcheck Command Live Test — PASS

Executed the exact fix command **inside the running container**:
```
$ docker exec knowledge-graph-mcp bash -c \
  'curl -sf --max-time 3 http://localhost:8004/sse; code=$?; \
   echo "CURL_EXIT_CODE:$code"; \
   [ $code -eq 0 ] || [ $code -eq 28 ]; \
   echo "HEALTHCHECK_RESULT:$?"'

event: endpoint
data: /messages/?session_id=3d970c4f76934bd28080e2d1d4103877

CURL_EXIT_CODE:28
HEALTHCHECK_RESULT:0
```

**The healthcheck command exits 0 — Docker will mark the container as `healthy`.** ✅

### 5. Downstream Impact Check — PASS

No services in `docker-compose.yml` depend on `knowledge-graph-mcp` with `condition: service_healthy`:
```
# service_healthy dependencies found:
postgres    ← used by auth-service, others
neo4j       ← used by knowledge-graph-builder, knowledge-graph-mcp
redis       ← used by knowledge-graph-builder, knowledge-graph-worker
knowledge-graph-builder ← used by knowledge-graph-mcp

knowledge-graph-mcp ← NOT used by any service as service_healthy dependency
```
No downstream cascade failures. ✅

### 6. Running Container Health Status

```json
{
  "Status": "unhealthy",
  "FailingStreak": 29,
  "Log": [{"ExitCode": -1, "Output": "Health check exceeded timeout (10s)..."}]
}
```

**Expected** — the running container was started from a branch that does NOT have the fix merged. The old `CMD curl -f http://localhost:8004/sse` healthcheck times out at 10s with ExitCode=-1.

Running container healthcheck config (old):
```json
{"Test": ["CMD", "curl", "-f", "http://localhost:8004/sse"], "Timeout": 10s}
```

---

## Pass Criteria Results

| Criterion | Result | Evidence |
|---|---|---|
| Fix uses CMD-SHELL accepting exit code 28 | ✅ PASS | Code diff verified |
| curl available in container image | ✅ PASS | Dockerfile line 5 |
| SSE endpoint returns HTTP 200 | ✅ PASS | `curl --max-time 3` → event:endpoint data |
| New healthcheck command exits 0 inside container | ✅ PASS | `HEALTHCHECK_RESULT:0` |
| No services blocked by mcp `service_healthy` | ✅ PASS | docker-compose.yml grep |
| Container shows `healthy` live | ⚠️ BLOCKED | PR #19 not yet merged; container still running old config |
| No `ExitCode=-1` after fix deployed | ⚠️ PENDING | Will clear on container recreate post-merge |

---

## Finding: Pre-Deployment State

The fix is on PR #19 (not yet merged to `develop`). The running `knowledge-graph-mcp` container was built from a branch where the docker-compose.yml has `CMD curl -f http://localhost:8004/sse` (ORA-228 fix, no timeout), which fails as expected.

**Required action to reach full live PASS:**
1. Merge PR #19 to `develop`
2. `git pull origin develop` in workspace
3. `docker compose up -d --force-recreate knowledge-graph-mcp`
4. Wait 15s (start_period) + 30s (first interval)
5. `docker compose ps` → should show `healthy`

---

## Conclusion

**Code verification: PASS.** The fix is correct, well-scoped, and has been validated by live execution of the healthcheck command inside the running container. No regressions introduced. PR #19 may be merged to `develop`.

Full live container health status (`healthy`) will be confirmed after deployment.
