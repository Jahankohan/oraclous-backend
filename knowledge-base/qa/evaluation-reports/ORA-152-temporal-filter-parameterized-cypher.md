---
title: "QA Evaluation Report — ORA-152: Temporal Filter Parameterized Cypher"
author: QA Evaluation Engineer
date: 2026-04-08
status: passed
source: ORA-139
target: fix/phase1/graph-dev/ora-139-temporal-filter-parameterize
---

# QA Evaluation Report — ORA-152
## Temporal Filter Parameterized Cypher (ORA-139)

**Date:** 2026-04-08
**Branch:** `fix/phase1/graph-dev/ora-139-temporal-filter-parameterize`
**Commit:** `b8c8e4b`
**Result:** PASSED ✅

---

## Summary

Validated the ORA-139 fix which refactors `compile_temporal_filter()` to return a
`Tuple[str, Dict[str, Any]]` instead of a plain `str`, moving all datetime ISO values
out of the Cypher clause string and into a parameterized dict.

---

## Validation Checklist

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `compile_temporal_filter()` returns `Tuple[str, Dict[str, Any]]` | ✅ PASS |
| 2 | No datetime ISO values in WHERE clause string (only `$tf_pit`, `$tf_vf_gte`, `$tf_vt_lte` refs) | ✅ PASS |
| 3 | All 5 `TestCompileTemporalFilter` tests pass | ✅ PASS (5/5) |
| 4 | `point_in_time` and `valid_from`/`valid_to` produce correct param keys | ✅ PASS |
| 5 | `current_only=True` fast-path returns empty params dict | ✅ PASS |

---

## Test Execution

### Pre-existing blocker
The SQLAlchemy `Table 'knowledge_graphs' already defined` error (tracked in ORA-147)
blocks full `pytest` collection of `test_temporal.py` via the `ChatRequest` import chain:

```
ChatRequest → retriever_factory → background_job_service → background_jobs → app.models.graph
```

### Isolation approach
Ran `TestCompileTemporalFilter` in isolation by patching `app.models.graph`,
`app.services.background_jobs`, and `app.schemas.chat_schemas` at the `sys.modules`
level before import, then exercising the 5 test cases directly against
`MultiTenantGraphRAGPipeline.compile_temporal_filter`.

### Results: 5/5 PASSED

```
PASS: test_current_only_produces_null_check
PASS: test_point_in_time_produces_range_clauses
PASS: test_empty_filter_returns_true
PASS: test_range_filter_includes_both_bounds
PASS: test_graph_id_not_injected_into_clause
```

---

## Code Review Observations

### Return Type (pipeline_service.py:1143)
```python
def compile_temporal_filter(
    self, temporal_filter: TemporalFilter
) -> Tuple[str, Dict[str, Any]]:
```
Confirmed correct signature.

### `current_only` fast-path
```python
if temporal_filter.current_only:
    return "r.valid_to IS NULL", {}
```
Returns empty params dict ✅

### `point_in_time` parameterization
```python
params["tf_pit"] = temporal_filter.point_in_time.isoformat()
clauses.append("(r.valid_from IS NULL OR r.valid_from <= datetime($tf_pit))")
clauses.append("(r.valid_to IS NULL OR r.valid_to > datetime($tf_pit))")
```
ISO value in `params`, Cypher param ref in clause ✅

### Range filter parameterization
```python
params["tf_vf_gte"] = temporal_filter.valid_from_gte.isoformat()
clauses.append("(r.valid_from IS NULL OR r.valid_from >= datetime($tf_vf_gte))")

params["tf_vt_lte"] = temporal_filter.valid_to_lte.isoformat()
clauses.append("(r.valid_to IS NULL OR r.valid_to <= datetime($tf_vt_lte))")
```
Both bounds parameterized correctly ✅

### Empty filter
```python
return (" AND ".join(clauses) if clauses else "true"), params
```
Returns `"true"` with empty params dict when no conditions set ✅

---

## Known Pre-existing Issue

**ORA-147** — SQLAlchemy `Table 'knowledge_graphs' already defined` MetaData conflict
blocks full `test_temporal.py` pytest collection. This is tracked separately and does
**not** affect the validity of the ORA-139 fix. The `TestCompileTemporalFilter` tests
themselves are logically correct and execute correctly in isolation.

---

## Verdict

**APPROVED for merge.** The ORA-139 fix correctly implements Cypher parameter
separation. All 5 target test cases pass. No regressions introduced in
`compile_temporal_filter` logic.
