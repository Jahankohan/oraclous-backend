---
id: TASK-002
title: "Generate per-level LLM summaries for Leiden community hierarchy"
story: STORY-001
assignee: ai-agent-specialist
reporter: reza
priority: critical
status: open
created: 2026-04-26
updated: 2026-04-26
blocked_by: [TASK-001]
blocks: [TASK-003]
wiki_refs: ["layer-1-knowledge-graph"]
estimated: "2d"
branch: "agent/STORY-001/TASK-002-leiden-summaries"
---

# TASK-002: Multi-Level Community LLM Summaries

## What

After TASK-001 persists the Leiden hierarchy, generate an LLM summary for each
`__Community__` node at each level. Summaries become more abstract at coarser levels.

1. **Add `_generate_level_summaries(graph_id)` to `analytics_service.py`**:
   - For each level (0 → 4), query all `__Community__` nodes at that level
   - For each community, retrieve its member `__Entity__` nodes (names + types)
   - Call LLM with the level-appropriate prompt (see below)
   - Write `summary` and `summary_level` fields to the `__Community__` node

2. **Prompt templates by level**:
   - Level 0 (finest): `"Summarize the entities and their relationships in this group: {entity_list}"`
   - Level 1: `"Summarize the themes and patterns across these sub-communities: {child_summaries}"`
   - Level 2+: `"What are the overarching topics and insights? {child_summaries}"`
   - Higher levels receive child summaries as input, not raw entity lists

3. **Chain summary generation in `background_jobs.py`**: After `detect_communities_task`
   completes, trigger summary generation as a chained Celery task (not a separate manual step).
   Use `task_a.apply_async(link=task_b.s())` pattern already used elsewhere in `background_jobs.py`.

4. **Staleness**: If community detection re-runs (e.g., new entities ingested), summaries
   must be regenerated. Clear `summary` field on stale communities before re-running.

**Files to modify:**
- `app/services/analytics_service.py` — add `_generate_level_summaries()`
- `app/services/background_jobs.py` — chain summary task after community detection

## Why

Community summaries are what the `COMMUNITY_SUMMARY` retriever (TASK-003) will serve
for global queries. Without summaries written to L1, there is nothing to retrieve.
Higher-level summaries synthesize lower-level ones to produce the thematic abstraction
that global search needs.

## Scope

**In scope:**
- Per-level summary generation using LLM
- Writing `summary` and `summary_level` to `__Community__` nodes
- Chaining as a Celery task after TASK-001's detection

**Out of scope (explicit):**
- Retrieval wiring (TASK-003)
- Changing the LLM provider or prompt engineering beyond the three-level template
- Async streaming (use synchronous LLM call per community; parallelize at Celery level if needed)

## Definition of Done

- [ ] Every `__Community__` node at every level has a non-empty `summary` field after task runs
- [ ] Level-0 summary references specific entity names; level-2+ summary is clearly more abstract
- [ ] Summary task is chained automatically after `detect_communities_task` — no manual trigger needed
- [ ] Stale `summary` fields are cleared before re-generation
- [ ] Unit test: level-0 prompt contains entity names; level-1+ prompt contains child summaries
- [ ] Reviewed by: qa-engineer (TASK-004)

## Output

- `code`: `app/services/analytics_service.py`, `app/services/background_jobs.py`

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

<!-- Not started. Blocked by TASK-001. -->
