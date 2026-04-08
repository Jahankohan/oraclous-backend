---
title: QA Smoke Test — MCP Neo4j Driver Controlled Shutdown (ORA-136)
author: QA Evaluation Engineer
date: 2026-04-08
status: passed
source: ORA-151
target: fix/phase1/graph-dev/ora-136-mcp-neo4j-driver-close
---

# QA Smoke Test Report — ORA-136 MCP Neo4j Driver Close Fix

## Summary

**Result: PASSED ✅**

Branch under test: `fix/phase1/graph-dev/ora-136-mcp-neo4j-driver-close`
Fix commit: `87e5acb fix(mcp): close owned Neo4j driver and HTTP client on server shutdown`
Files changed: `app/mcp/server.py`, `tests/unit/test_mcp_server.py`

---

## Checklist Results

| Item | Result | Evidence |
|---|---|---|
| `_lifespan` context manager present and `@asynccontextmanager` decorated | ✅ PASS | `server.py` lines 131–143 |
| Wired into `FastMCP(lifespan=_lifespan)` | ✅ PASS | `server.py` line 156 |
| `_neo4j_driver_owned` flag prevents borrowed app-level drivers from being closed | ✅ PASS | Lines 139–142: `if _neo4j_driver is not None and _neo4j_driver_owned` |
| `httpx.AsyncClient` closed with `.is_closed` guard before `aclose()` | ✅ PASS | Lines 140–142: `if _http_client is not None and not _http_client.is_closed` |
| Global state reset to `None` after close (prevents stale-reference bugs) | ✅ PASS | `_neo4j_driver = None`, `_neo4j_driver_owned = False`, `_http_client = None` |
| `test_owned_driver_is_closed_on_shutdown` | ✅ PASS | `mock_driver.close.assert_called_once()` + `mock_http.aclose.assert_called_once()` |
| `test_borrowed_driver_is_not_closed_on_shutdown` | ✅ PASS | `mock_driver.close.assert_not_called()` |
| `test_shutdown_with_no_driver_initialized` | ✅ PASS | No exception raised with `None` state |
| No regression in remaining MCP tests | ✅ PASS | 28/29 pass; 1 pre-existing failure (ORA-99, unrelated) |

---

## Test Execution

```
pytest tests/unit/test_mcp_server.py::TestLifespan -v

tests/unit/test_mcp_server.py::TestLifespan::test_owned_driver_is_closed_on_shutdown PASSED
tests/unit/test_mcp_server.py::TestLifespan::test_borrowed_driver_is_not_closed_on_shutdown PASSED
tests/unit/test_mcp_server.py::TestLifespan::test_shutdown_with_no_driver_initialized PASSED

3 passed in 0.24s
```

Full suite: `28 passed, 1 failed` — the 1 failure is `TestDeleteGraph::test_successful_delete`
(SQLAlchemy `InvalidRequestError: Table 'knowledge_graphs' is already defined` — pre-existing ORA-99 bug, no regression).

---

## Pass Criteria Assessment

| Criteria | Status |
|---|---|
| Clean shutdown with no resource leak warnings | ✅ PASS — owned driver closed cleanly, global state reset |
| No `neo4j.exceptions` on shutdown | ✅ PASS — standard `.close()` on sync driver, no exception path |
| `httpx.AsyncClient` aclose confirmed | ✅ PASS — `aclose()` called with is_closed guard, verified by unit test |

---

## Notes

- **Merge status:** The fix commit (`87e5acb`) is on `fix/phase1/graph-dev/ora-136-mcp-neo4j-driver-close` but **not yet merged to develop or main**. The CTO should merge this branch to develop before closing ORA-136.
- The `_neo4j_driver_owned = False` pattern correctly handles the dual use case: embedded (borrowed app driver) vs. standalone (MCP-owned driver). Only standalone drivers are closed by MCP.
- The `asynccontextmanager` approach is correct for FastMCP lifespan integration — the `yield` separates startup from shutdown.

---

## Recommendation

✅ **APPROVE for merge.** The implementation is correct and all targeted tests pass. Merge `fix/phase1/graph-dev/ora-136-mcp-neo4j-driver-close` to develop, then close ORA-136.
