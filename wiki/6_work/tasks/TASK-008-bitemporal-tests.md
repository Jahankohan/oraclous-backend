---
id: TASK-008
title: "QA: Test suite for bitemporal properties, extraction, and temporal query modes"
story: STORY-002
assignee: qa-engineer
reporter: reza
priority: critical
status: open
created: 2026-04-26
updated: 2026-04-26
blocked_by: [TASK-005, TASK-006, TASK-007]
blocks: []
wiki_refs: []
estimated: "2d"
branch: "agent/STORY-002/TASK-008-bitemporal-tests"
---

# TASK-008: STORY-002 Test Suite

## What

Write and run all tests for the bitemporal tracking work. Begins when TASK-005, TASK-006,
and TASK-007 are all `in-review`.

### Unit tests:

1. **Properties stored independently**: Create a test entity via pipeline; assert
   `event_time != ingestion_time` when source text contains a specific date. Assert
   both are stored as separate Neo4j properties.

2. **Null event_time when no date in text**: Ingest a document with no date mentions;
   assert extracted relationships have `event_time = null`, not a substituted value.

3. **Migration idempotency**: Run the migration twice on the same dataset; assert no
   duplicate nodes, no property overwrite errors, no data loss.

4. **point_in_time filter**: Create two relationships — one valid in 2020-2022, one
   valid in 2023-present. Query with `point_in_time, at=2021-06-01`; assert only the
   2020-2022 relationship is returned.

5. **knowledge_as_of filter**: Create two facts — one ingested 2026-04-01, one ingested
   2026-04-20. Query with `knowledge_as_of, at=2026-04-10`; assert only the April 1 fact
   is returned.

6. **changes_since filter**: Same two facts. Query with `changes_since=2026-04-15`;
   assert only the April 20 fact is returned.

7. **Backward compatibility**: Chat request with no `temporal_mode` returns same results
   as before (regression check against a baseline).

### Integration test:

8. **End-to-end temporal extraction and query**: Ingest a test document stating "Alice was
   CEO of Acme from 2019 to 2023". Send `point_in_time` chat query for Acme's CEO in 2021.
   Assert Alice is returned. Send same query for 2024. Assert Alice is NOT returned (or is
   returned with context that the relationship ended).

### Regression:

9. Run all existing chat and memory service tests; assert no regressions from TASK-005
   model changes.

## Why

Bitemporal logic is subtle and error-prone. Incorrect filter logic (off-by-one on date
comparisons, null handling) produces silently wrong answers that are hard to detect in
production. Thorough tests here prevent degraded temporal accuracy.

## Scope

**In scope:**
- Unit tests 1-7 listed above
- Integration test 8
- Regression run (test 9)

**Out of scope (explicit):**
- Performance testing of temporal queries (STORY-008)
- UI testing

## Definition of Done

- [ ] All 9 test scenarios pass
- [ ] No existing chat or memory service tests regressed
- [ ] All STORY-002 acceptance criteria verified and marked complete in STORY-002
- [ ] Reviewed by: reza

## Output

- `test`: additions to memory service unit tests + chat service integration tests

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

<!-- Not started. Blocked by TASK-005, TASK-006, TASK-007. -->
