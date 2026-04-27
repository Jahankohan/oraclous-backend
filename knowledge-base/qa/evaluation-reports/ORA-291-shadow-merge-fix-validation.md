---
title: "ORA-291 — QA Validation: Shadow MERGE Fix (ORA-287 / ORA-290)"
author: QA Evaluation Engineer
date: 2026-04-11
status: PASSED
source: tests/integration/test_shadow_merge_ora291.py
target: app/services/graph_node_service.py — update_graph() MERGE shadow fix
---

# ORA-291 QA Validation Report — Shadow MERGE Fix

## Summary

**Result: 14/14 PASSED — All three validation scenarios passed.**

The ORA-287 / ORA-290 fix (replace `OPTIONAL MATCH` with `MERGE` in `GraphNodeService.update_graph()`) is validated and correct. The shadow node is now reliably created when absent, updated without clobbering existing properties, and the fail-closed regression is preserved.

---

## Context

| Field | Value |
|---|---|
| Fix branch | `fix/phase1/graph-dev/ora-287-shadow-merge` |
| Merge commit | `553ac54f` |
| Merged to | `develop` |
| Test environment | Local Neo4j `neo4j://localhost:7687` |
| Python | 3.12.8 |
| pytest | 8.4.1 |
| Run date | 2026-04-11 |

---

## Test Results

```
============================= test session starts ==============================
platform darwin -- Python 3.12.8, pytest-8.4.1
rootdir: oraclous-data-studio/knowledge-graph-builder
configfile: pytest.ini

tests/integration/test_shadow_merge_ora291.py  [14/14]

============================ 14 passed in 0.58s ============================
```

---

## Scenario Breakdown

### Scenario 1 — Bootstrap Path (shadow absent on new graph) ✅ 5/5

Tests that `update_graph(federatable=True)` on a **brand-new graph** (no pre-existing shadow node) correctly creates the shadow node via `MERGE`.

| Test | Result | What Was Verified |
|---|---|---|
| `test_shadow_absent_before_update` | PASS | Pre-condition: new graph has no shadow node |
| `test_update_federatable_creates_shadow_node` | PASS | `MERGE` creates shadow node when absent |
| `test_shadow_node_has_correct_federatable_flag` | PASS | `shadow.federatable = true` after bootstrap |
| `test_shadow_node_namespace_is_system` | PASS | `shadow.namespace = '__system__'` |
| `test_shadow_node_carries_graph_id` | PASS | `shadow.graph_id` matches the graph |

**Before the fix:** `OPTIONAL MATCH` would silently no-op — shadow never created → subsequent federation query returned 400 with misleading error.

**After the fix:** Shadow node created atomically on first `federatable=true` update. Federation query returns 200.

---

### Scenario 2 — Update Path (shadow already exists) ✅ 5/5

Tests that `update_graph(federatable=True)` on a graph **that already has a shadow node** only updates `federatable` — does not overwrite `owner_user_id`, `graph_name`, or any other shadow property.

| Test | Result | What Was Verified |
|---|---|---|
| `test_update_sets_federatable_true` | PASS | `federatable` flips to `true` |
| `test_update_preserves_owner_user_id` | PASS | `owner_user_id` not overwritten |
| `test_update_preserves_graph_name` | PASS | `graph_name` not overwritten |
| `test_update_preserves_arbitrary_sentinel_property` | PASS | Extra properties survive `MERGE+SET` |
| `test_only_one_shadow_node_after_repeated_updates` | PASS | 3× updates → still exactly 1 shadow node (idempotent) |

**Key correctness property confirmed:** The Cypher pattern `MERGE (shadow:Graph {graph_id: …, namespace: "__system__"}) SET shadow.federatable = $federatable` correctly uses property-level SET — not `SET shadow = {…}` — so non-federatable shadow properties are never clobbered.

---

### Scenario 3 — Regression (fail-closed preserved) ✅ 4/4

Tests that disabling federation (`federatable=false`) correctly reflects in the shadow node, and that updates without a `federatable` kwarg do not touch the shadow node at all.

| Test | Result | What Was Verified |
|---|---|---|
| `test_shadow_federatable_false_after_disable` | PASS | `shadow.federatable = false` after disable |
| `test_no_shadow_sync_when_federatable_not_in_payload` | PASS | Non-federatable updates leave shadow untouched |
| `test_graph_node_federatable_also_set_to_false` | PASS | Primary graph node also reflects `false` |
| `test_shadow_merge_does_not_overwrite_on_disable` | PASS | `owner_user_id` preserved after disable |

**Fail-closed guarantee confirmed intact:** Setting `federatable=false` sets `shadow.federatable=false` correctly. Federation queries against non-federatable graphs will still fail with 400 as designed.

---

## Acceptance Criteria

| Criterion | Status |
|---|---|
| Scenario 1 — Bootstrap path passes | ✅ PASS |
| Scenario 2 — Update path, no shadow property regression | ✅ PASS |
| Scenario 3 — Fail-closed still enforced | ✅ PASS |
| No test isolation leakage (cleanup fixture verified) | ✅ PASS |

---

## QA Gate: CLEARED

The shadow MERGE fix is validated. ORA-287 / ORA-290 can be considered complete.

Parent bug [ORA-251](/ORA/issues/ORA-251) (federatable flag not synced to ReBAC shadow node) has its implementation path fully validated:
- [ORA-252](/ORA/issues/ORA-252) — rebac_service federatable sync ✅ (previously validated)
- [ORA-287](/ORA/issues/ORA-287) → [ORA-290](/ORA/issues/ORA-290) — shadow MERGE fix ✅ (this report)

No new bugs found. No regressions detected.
