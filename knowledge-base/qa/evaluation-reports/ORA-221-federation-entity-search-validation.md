---
title: "QA Validation Report — Federation Entity Search Cypher Fix (ORA-221)"
author: QA Evaluation Engineer
date: 2026-04-09
status: PASSED
source: commit b139fff4 (merged to develop)
target: ORA-217 / ORA-221
---

# QA Validation Report — Federation Entity Search Fix

**Task:** [ORA-221](/ORA/issues/ORA-221)
**Fix:** [ORA-217](/ORA/issues/ORA-217)
**Commit:** `b139fff4` (merged to `develop` via PR #14)
**Date:** 2026-04-09
**Verdict:** ✅ **PASSED — Fix is correct and all tests green**

---

## Root Cause (ORA-217)

Neo4j 5.23+ rejects Cypher queries where `CALL {}` subqueries are joined at the top level with `UNION ALL`. The old `_execute_entity_union` in `federation_service.py` generated:

```cypher
CALL {
  MATCH (e:__Entity__) WHERE e.graph_id = $gid_0 ...
  RETURN ... LIMIT $limit
}
UNION ALL
CALL {
  MATCH (e:__Entity__) WHERE e.graph_id = $gid_1 ...
  RETURN ... LIMIT $limit
}
RETURN entity_id, name, type, source_graph_id
```

This caused `SyntaxError: Query cannot conclude with CALL` on every federated entity search.

---

## Fix Verification

The fix restructures to a single outer CALL wrapping all UNION ALL branches:

```cypher
CALL {
  MATCH (e:__Entity__) WHERE e.graph_id = $gid_0 ...
  RETURN ... LIMIT $limit
  UNION ALL
  MATCH (e:__Entity__) WHERE e.graph_id = $gid_1 ...
  RETURN ... LIMIT $limit
}
RETURN entity_id, name, type, source_graph_id
```

**Code review confirmed:**
- No per-branch `CALL {}` wrappers ✅
- Single outer `CALL {}` wrapping all UNION ALL branches ✅
- `$gid_N AS source_graph_id` → `e.graph_id AS source_graph_id` (safe: WHERE clause enforces equality) ✅
- Params structure unchanged — `search_term`, `limit`, `gid_N` per graph ✅

---

## Test Results

### Federation Service Unit Tests (ORA-217 regression suite)

```
tests/unit/test_federation_service.py::test_validate_rejects_cross_tenant_graph              PASSED
tests/unit/test_federation_service.py::test_validate_rejects_non_federatable_graph           PASSED
tests/unit/test_federation_service.py::test_validate_rejects_missing_graph                   PASSED
tests/unit/test_federation_service.py::test_validate_rejects_too_many_graphs                 PASSED
tests/unit/test_federation_service.py::test_validate_passes_for_all_owned_and_federatable    PASSED
tests/unit/test_federation_service.py::test_validate_uses_owner_user_id_not_user_id          PASSED
tests/unit/test_federation_service.py::test_validate_matches_system_namespace                PASSED
tests/unit/test_federation_service.py::test_entity_union_passes_graph_ids_as_params          PASSED
tests/unit/test_federation_service.py::test_federated_query_returns_source_attribution       PASSED
tests/unit/test_federation_service.py::test_same_as_deduplication_produces_link_for_matching PASSED
tests/unit/test_federation_service.py::test_store_same_as_links_is_awaited                  PASSED
tests/unit/test_federation_service.py::test_same_as_no_link_for_same_graph                  PASSED
tests/unit/test_federation_service.py::test_same_as_no_link_for_different_types             PASSED
tests/unit/test_federation_service.py::test_entity_union_cypher_uses_single_outer_call      PASSED  ← ORA-217
tests/unit/test_federation_service.py::test_entity_union_cypher_structure_single_call_wrapping PASSED  ← ORA-217
tests/unit/test_federation_service.py::test_federated_query_request_rejects_duplicate_graph_ids PASSED
tests/unit/test_federation_service.py::test_federated_query_request_rejects_single_graph    PASSED
tests/unit/test_federation_service.py::test_federated_query_request_rejects_too_many_graphs PASSED

Result: 18/18 passed
```

### Full KG-Builder Unit Suite (Regression)

```
Total: 713 passed, 7 xfailed — 0 failures, 0 regressions
```

---

## Acceptance Criteria

| Criterion | Status |
|---|---|
| Cypher no longer uses per-branch `CALL {}` wrappers | ✅ Confirmed in code |
| Single outer `CALL {}` wraps all UNION ALL branches | ✅ Confirmed in code |
| Two ORA-217 regression tests added and passing | ✅ 18/18 tests pass |
| No regressions in existing federation tests | ✅ All prior 16 tests still pass |
| Full kg-builder unit suite passes | ✅ 713/713 |

---

## Note on Live Endpoint Testing

Live stack (Docker containers + Neo4j 5.23+) is not running in local QA environment. Unit-level validation confirms the Cypher template is structurally correct per Neo4j 5.23+ requirements. The two new regression tests (`test_entity_union_cypher_uses_single_outer_call`, `test_entity_union_cypher_structure_single_call_wrapping`) provide high confidence the fix resolves ORA-217. Full live smoke test can be performed in staging when Neo4j environment is available.

---

## Decision

**✅ QA APPROVED** — Cypher fix is structurally correct, all 18 federation tests pass, 713/713 unit tests pass with zero regressions. ORA-217 fix is validated.
