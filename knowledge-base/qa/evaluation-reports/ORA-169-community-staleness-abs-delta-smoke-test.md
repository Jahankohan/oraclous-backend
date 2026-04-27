---
title: "QA Smoke Test Report — ORA-137 Community Staleness Absolute Delta Fix"
author: "QA Evaluation Engineer"
date: "2026-04-09"
status: "pass"
source: "ORA-169"
target: "fix(background_jobs): ORA-137 — use abs() for community staleness delta"
branch: "fix/phase1/graph-dev/ora-137-community-staleness-absolute-delta"
merged_to: "develop"
pr: "https://github.com/Jahankohan/oraclous-backend/pull/4"
---

# QA Smoke Test Report — ORA-137 Community Staleness Absolute Delta Fix

## Summary

**Result: PASS** — All 4 acceptance criteria met. No regressions detected.

| Test Area | Result | Tests |
|---|---|---|
| Regression fix (negative delta) | ✅ PASS | `test_negative_delta_above_threshold_marks_stale` |
| Growth path (positive delta) | ✅ PASS | `test_positive_delta_above_threshold_marks_stale` |
| Below-threshold no-op | ✅ PASS | `test_delta_below_threshold_not_stale` |
| Zero-baseline guard | ✅ PASS | `test_zero_entity_count_at_detection_no_zero_division_error` |
| Full community detection suite | ✅ PASS | 24/24 tests |
| background_jobs unit suite | ✅ PASS | 24/24 tests |

## Fix Verification

**File:** `knowledge-graph-builder/app/services/background_jobs.py:750`

Before (buggy):
```python
staleness = new_delta / row[0]
```

After (fix):
```python
staleness = abs(new_delta) / max(row[0], 1)
```

The root cause was that when `entity_delta_since_detection` is negative (entities deleted), `new_delta` becomes negative, and without `abs()` the staleness ratio was also negative — never exceeding the 0.10 threshold. Communities were therefore never marked stale after entity deletions. The fix applies `abs()` to ensure both positive (growth) and negative (deletion) deltas trigger staleness correctly.

The `max(row[0], 1)` guard provides defence-in-depth against `ZeroDivisionError` when `entity_count_at_detection = 0`, though the outer condition `if row and row[0] and row[0] > 0` already prevents this path.

## Test Execution

**Command:**
```
python3 -m pytest tests/unit/test_community_detection.py::TestMaybeTriggerCommunityDetectionStaleness -v
```

**Output:**
```
collected 4 items

TestMaybeTriggerCommunityDetectionStaleness::test_negative_delta_above_threshold_marks_stale PASSED
TestMaybeTriggerCommunityDetectionStaleness::test_positive_delta_above_threshold_marks_stale PASSED
TestMaybeTriggerCommunityDetectionStaleness::test_delta_below_threshold_not_stale            PASSED
TestMaybeTriggerCommunityDetectionStaleness::test_zero_entity_count_at_detection_no_zero_division_error PASSED

4 passed in 0.24s
```

**Regression suite (full community detection):**
```
python3 -m pytest tests/unit/test_community_detection.py -v
24 passed in 0.12s
```

**Regression suite (background_jobs):**
```
python3 -m pytest tests/unit/test_background_jobs.py -v
24 passed in 0.11s
```

## Acceptance Criteria — Verification

| Criterion | Status | Evidence |
|---|---|---|
| After deleting entities below baseline, subsequent ingest marks communities stale | ✅ | `test_negative_delta_above_threshold_marks_stale` passes |
| After adding entities above baseline (>10%), communities marked stale | ✅ | `test_positive_delta_above_threshold_marks_stale` passes |
| Small changes (<10%) in either direction do NOT mark communities stale | ✅ | `test_delta_below_threshold_not_stale` passes |
| Zero baseline guard — no ZeroDivisionError | ✅ | `test_zero_entity_count_at_detection_no_zero_division_error` passes |

## Notes

- Unit test collection was previously blocked by SQLAlchemy double-registration (ORA-111). This did not affect the current test run — all tests collected and passed cleanly.
- Manual end-to-end verification (ingest → delete → ingest cycle against a running instance) was not performed as services are not deployed locally. Unit tests fully cover the staleness logic path with I/O mocked.
