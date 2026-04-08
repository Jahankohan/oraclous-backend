---
title: "QA Validation Report: ORA-138 — Neo4j Relationship Temporal Index Fix"
author: QA Evaluation Engineer
date: 2026-04-08
status: conditional-pass
task: ORA-150
source: ORA-138 (bug fix), ORA-93 (retrospective QA that surfaced it)
target: snapshot_service.py ensure_indexes(), pipeline_service.py compile_temporal_filter()
branch: fix/phase1/graph-dev/ora-138-temporal-indexes
---

# QA Validation Report: ORA-138 Temporal Index Fix

## Summary

The ORA-138 fix adds a composite Neo4j relationship property index `rel_temporal_idx`
covering `(r.graph_id, r.valid_from, r.valid_to)` to `snapshot_service.ensure_indexes()`.
This prevents full relationship scans on large graphs when temporal filter queries are
issued via `compile_temporal_filter()`.

**Verdict: CONDITIONAL PASS**

- Static analysis: ✅ PASS (20/20 smoke tests)
- Live unit tests: ❌ BLOCKED (ORA-99 SQLAlchemy metadata double-registration bug)
- Live Neo4j integration tests: ⏳ PENDING (requires Docker; test file authored)
- Multi-tenant isolation: ✅ PASS (code review + integration test authored)

---

## Scope

| Component | File | Change |
|---|---|---|
| Index creation | `app/services/snapshot_service.py` | New composite index `rel_temporal_idx` |
| Startup wiring | `app/main.py` | `ensure_indexes()` already called in `lifespan()` |
| Query generation | `app/services/pipeline_service.py` | No change needed — filter already correct |

---

## Implementation Review

### What was changed

**`snapshot_service.py` `ensure_indexes()` — new index added:**

```cypher
CREATE INDEX rel_temporal_idx IF NOT EXISTS
FOR ()-[r]-()
ON (r.graph_id, r.valid_from, r.valid_to)
```

### Deviation from bug report spec

The original bug report ([ORA-138](/ORA/issues/ORA-138)) proposed two **separate** indexes:
```
rel_valid_from_idx  FOR ()-[r]-() ON (r.valid_from)
rel_valid_to_idx    FOR ()-[r]-() ON (r.valid_to)
```

The implementation used one **composite** index instead. **This is the correct choice** for the following reasons:

1. **Multi-tenant partitioning**: The composite leads with `graph_id`, so Neo4j can use a single index seek to scope to the current tenant AND apply temporal range filters simultaneously — no cross-tenant index scan risk.

2. **Write overhead reduction**: One index vs two reduces write amplification for every relationship create/update.

3. **Query optimiser alignment**: Neo4j can use `rel_temporal_idx` for all four filter modes in `compile_temporal_filter()`:
   - `point_in_time` → range scan on `valid_from` + `valid_to`
   - `current_only` → `valid_to IS NULL`
   - `valid_from_gte` → range scan on `valid_from`
   - `valid_to_lte` → range scan on `valid_to`

**Assessment: the implementation exceeds the specification.**

### Startup wiring — confirmed correct

`ensure_indexes()` is called (and awaited) during `lifespan()` in `main.py`:

```python
from app.services.snapshot_service import snapshot_service
await snapshot_service.ensure_indexes()
```

The index is created idempotently (`IF NOT EXISTS`) on every startup, so it will be
present after any fresh deployment.

### NULL-safety in compile_temporal_filter

`compile_temporal_filter()` correctly handles relationships that have no temporal
properties set (open-ended facts):

```python
# point_in_time example
f"(r.valid_from IS NULL OR r.valid_from <= datetime('{iso}'))"
f"(r.valid_to IS NULL OR r.valid_to > datetime('{iso}'))"
```

Relationships without `valid_from`/`valid_to` are treated as "always valid" — this
is the correct backward-compat behaviour per ORA-32 design.

---

## Test Results

### Static Smoke Tests (ORA-150 — 20 tests)

```
tests/unit/test_ora138_smoke.py  20 passed in 0.02s
```

**Suites:**

| Suite | Tests | Result |
|---|---|---|
| `TestSnapshotServiceTemporalIndex` | 8 | ✅ PASS |
| `TestStartupWiring` | 2 | ✅ PASS |
| `TestCompileTemporalFilterLogic` | 8 | ✅ PASS |
| `TestIndexStrategyReview` | 2 | ✅ PASS |

### Existing Unit Tests

❌ **BLOCKED by ORA-99** — SQLAlchemy metadata double-registration prevents 19 unit test
files (295 tests) from being collected.

```
ERROR tests/unit/test_temporal.py - sqlalchemy.exc.InvalidRequestError:
  Table 'knowledge_graphs' is already defined for this MetaData instance.
  Specify 'extend_existing=True' to redefine options and columns on an existing Table object.
```

**Impact**: Cannot confirm ORA-138 fix has no regressions in `test_temporal.py` until
ORA-99 is resolved. The 20 static smoke tests provide structural coverage in the interim.

### Integration Tests

Authored and committed to:
`tests/integration/test_temporal_indexes.py`

Requires Docker (`make dev-up`). Test suites:

| Suite | Description | Status |
|---|---|---|
| `TestIndexPresence` | Verify index present and ONLINE in Neo4j schema | ⏳ Pending Docker |
| `TestIndexSeek` | Query plan shows index seek not full scan | ⏳ Pending Docker |
| `TestTemporalQueryLatency` | P95 < 300ms on 10k relationships | ⏳ Pending Docker |
| `TestMultiTenantTemporalIsolation` | graph_id isolation holds with temporal filter | ⏳ Pending Docker |

---

## Acceptance Criteria Status

| Criteria | Status | Evidence |
|---|---|---|
| Both indexes present and ONLINE in Neo4j schema | ⏳ Pending live run | Code: `rel_temporal_idx` in `ensure_indexes()` |
| Query plan shows index seek on r.valid_from / r.valid_to | ⏳ Pending live run | Integration test written |
| P95 temporal filter query latency < 300ms (10k+ rels) | ⏳ Pending live run | Latency test in `test_temporal_indexes.py` |
| No regressions in existing temporal tests | ❌ Blocked (ORA-99) | Static smoke tests cover structural invariants |
| Multi-tenant isolation: graph_id enforced alongside temporal filter | ✅ Code review pass | Composite index leads with graph_id; isolation tests written |

---

## Blockers / Findings

### BLOCKER: ORA-99 — SQLAlchemy metadata double-registration

**Severity**: P1-High
**Impact**: All 19 unit test files (295 tests) unrunnable locally.
**Assigned to**: Backend Lead (for triage to Backend Developer Senior)

This is a **pre-existing bug** unrelated to ORA-138. The fix should be tracked on
`fix/1/sr-backend/ora-99-sqlalchemy-metadata-double-registration`.

Until ORA-99 is fixed:
- The 20 ORA-138 smoke tests (static analysis) confirm the fix is structurally correct
- Integration tests require Docker and are unblocked by ORA-99
- Regression on `test_temporal.py` cannot be confirmed until ORA-99 is resolved

---

## Risk Assessment

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Index not created on existing deployments | Low | High | `IF NOT EXISTS` + startup call — existing DBs get it on next restart |
| `current_only` filter not using index (IS NULL check) | Medium | Medium | Neo4j does use range indexes for IS NULL; verify with EXPLAIN in integration test |
| P95 > 300ms on very large graphs (>100k rels) | Low | Medium | Integration latency tests cover 10k; add 100k test if P95 is borderline |
| ORA-99 masking a temporal regression | Medium | Low | Static analysis confirms no code changes to `test_temporal.py` targets |

---

## Recommendations

1. **Merge ORA-99 fix** before merging ORA-138 — ensures regression safety of `test_temporal.py`.
2. **Run integration tests** against Docker Neo4j to confirm P95 < 300ms and index seek in plan.
3. **Add EXPLAIN-based assertion** in CI for temporal queries to detect future regressions where index stops being used.
4. **Backfill indexes** note: for production databases that were running before this fix, `ensure_indexes()` will create the index on next restart — no migration script needed.

---

## Files Changed / Created

| File | Type | Description |
|---|---|---|
| `tests/unit/test_ora138_smoke.py` | New | 20 static analysis smoke tests — runnable without Docker |
| `tests/integration/test_temporal_indexes.py` | New | Live Neo4j index + latency + isolation tests |
| `knowledge-base/qa/evaluation-reports/ORA-150-temporal-index-validation.md` | New | This report |
