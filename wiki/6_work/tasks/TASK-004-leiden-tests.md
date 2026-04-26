---
id: TASK-004
title: "QA: Test suite for Leiden hierarchy, summaries, and community retriever"
story: STORY-001
assignee: qa-engineer
reporter: reza
priority: critical
status: open
created: 2026-04-26
updated: 2026-04-26
blocked_by: [TASK-001, TASK-002, TASK-003]
blocks: []
wiki_refs: []
estimated: "2d"
branch: "agent/STORY-001/TASK-004-leiden-tests"
---

# TASK-004: STORY-001 Test Suite

## What

Write and run all tests for the Leiden community detection work. This task begins only
after TASK-001, TASK-002, and TASK-003 are all `in-review`.

### Unit tests (add to existing analytics test suite):

1. **Leiden hierarchy structure**: Build a small igraph (10 nodes, known edges), run
   leidenalg at 2 resolutions, assert: (a) communities are non-empty, (b) every node
   belongs to exactly one community per level, (c) `PARENT_OF` edges exist between levels

2. **Parent-child mapping correctness**: Given a synthetic partition where community A
   (coarse) contains all members of communities X and Y (fine), assert the mapping
   correctly assigns X and Y as children of A

3. **Summary prompt content**: Mock the LLM call; assert level-0 prompt contains entity
   names, level-1 prompt contains child summaries (not raw entity lists), level-2 prompt
   is more abstract than level-1

4. **Global query routing**: Assert that query `"what are the main themes across all documents?"`
   routes to `COMMUNITY_SUMMARY` retriever; assert query `"who is John Smith?"` routes
   to vector/hybrid retriever

### Integration tests (add to existing integration suite):

5. **End-to-end community detection → summary → retrieval**: Ingest 3 test documents
   into a test graph; trigger community detection; assert `__Community__` nodes exist
   with `summary` populated at each level; send a global chat query; assert response
   references community-level insights

6. **Staleness: re-run after new ingestion**: After initial community detection, ingest
   a new document, re-trigger detection; assert stale communities are replaced, summaries
   are regenerated

### Regression:

7. Run all existing analytics and community detection tests; assert none regressed

## Why

TASK-001–003 touch core retrieval infrastructure. A regression here would silently
degrade all chat quality. The test suite also validates STORY-001's acceptance criteria
before marking the story `done`.

## Scope

**In scope:**
- Unit tests listed above (items 1-4)
- Integration tests listed above (items 5-6)
- Regression run of existing community test suite (item 7)

**Out of scope (explicit):**
- Performance benchmarking (STORY-008)
- Frontend testing

## Definition of Done

- [ ] All 7 test scenarios pass with no failures
- [ ] No existing analytics or community detection tests regressed
- [ ] All STORY-001 acceptance criteria verified (checklist in STORY-001 marked complete)
- [ ] Test run output attached to this task's Notes section
- [ ] Reviewed by: reza

## Output

- `test`: additions to existing analytics test file + integration test file

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

<!-- Not started. Blocked by TASK-001, TASK-002, TASK-003. -->
