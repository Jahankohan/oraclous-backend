---
id: TASK-001
title: "Switch community detection from GDS Louvain to leidenalg with multi-resolution hierarchy"
story: STORY-001
assignee: backend-developer
reporter: reza
priority: critical
status: in-progress
created: 2026-04-26
updated: 2026-04-26
blocked_by: []
blocks: [TASK-002, TASK-003]
wiki_refs: ["layer-1-knowledge-graph"]
estimated: "3-4d"
branch: "agent/STORY-001/TASK-001-leiden-algorithm"
---

# TASK-001: Switch Community Detection to leidenalg

## What

Replace the GDS Louvain call in `app/services/analytics_service.py` with a Python-side
`leidenalg` run on an extracted igraph object. Run Leiden at 5 resolution levels
(0.25, 0.5, 1.0, 2.0, 4.0) to produce a parent-child community hierarchy. Persist the
hierarchy as `__Community__` nodes with `PARENT_OF` edges in Neo4j.

Concretely:

1. **Extract igraph from Neo4j**: Query `__Entity__` nodes and their relationship edges
   for the target `graph_id`. Build an `igraph.Graph.TupleList` from the result.

2. **Run multi-resolution Leiden**:
   ```python
   import igraph as ig
   import leidenalg

   partitions = []
   for resolution in [0.25, 0.5, 1.0, 2.0, 4.0]:
       partition = leidenalg.find_partition(
           g, leidenalg.RBConfigurationVertexPartition,
           resolution_parameter=resolution
       )
       partitions.append((resolution, partition))
   ```

3. **Build parent-child mapping**: Communities at coarser resolutions (lower value)
   are parents of communities at finer resolutions. Determine parent by majority
   membership: which coarser community contains the most members of each finer community.

4. **Persist to Neo4j**:
   - Create `__Community__` nodes with fields: `id`, `graph_id`, `level` (0=finest),
     `resolution`, `entity_count`, `member_ids` (list), `weight`
   - Create `PARENT_OF` edges between community levels
   - Update `__Entity__` nodes: set `community_id` to their level-0 community
   - Delete stale `__Community__` nodes from previous runs before writing new ones

5. **Update `background_jobs.py`**: The `detect_communities_task` Celery task calls this
   new logic. Preserve existing task signature and retry policy.

**Files to modify:**
- `app/services/analytics_service.py` — replace `_detect_communities_gds()` method
- `app/services/background_jobs.py` — update `detect_communities_task`

**New dependency already in requirements.txt:** `leidenalg`, `igraph` (verify both present)

## Why

`leidenalg` is already installed but ignored; GDS Louvain produces flat communities.
The hierarchy produced by this task is what TASK-002 (LLM summaries) and TASK-003
(retrieval wiring) depend on. Without the correct hierarchy in L1, global search
cannot be implemented.

## Scope

**In scope:**
- Replacing the Louvain call with leidenalg on extracted igraph
- Persisting 5-level hierarchy as `__Community__` nodes + `PARENT_OF` edges
- Updating `__Entity__` nodes with `community_id` (level-0 assignment)
- Updating the Celery task to call new logic

**Out of scope (explicit):**
- LLM summary generation (TASK-002)
- Retrieval path changes (TASK-003)
- Per-tenant resolution parameter tuning (use global defaults)
- Changing `__Community__` node schema fields beyond adding `level`, `resolution`, `PARENT_OF`

## Definition of Done

- [ ] `detect_communities_task` runs to completion without error on a test graph with 100+ entities
- [ ] `__Community__` nodes in Neo4j have `level` field (0 = finest) and `PARENT_OF` edges link levels
- [ ] Every `__Entity__` node has `community_id` pointing to its level-0 `__Community__`
- [ ] No GDS Louvain call remains in `analytics_service.py`
- [ ] Unit test: Leiden on a 10-node igraph produces valid partition with non-empty communities
- [ ] Unit test: parent-child mapping — each finer community has exactly one parent at the next coarser level
- [ ] Reviewed by: qa-engineer (TASK-004)

## Output

- `code`: `app/services/analytics_service.py`, `app/services/background_jobs.py`
- `test`: unit tests in existing analytics test suite

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

### 2026-04-26 — backend-developer — open → in-progress

Starting implementation. Branch: `agent/STORY-001/TASK-001-leiden-algorithm`.
Switching analytics_service.py from GDS Louvain to leidenalg at 5 resolutions.
Work in progress — leidenalg library already present in requirements.txt.
