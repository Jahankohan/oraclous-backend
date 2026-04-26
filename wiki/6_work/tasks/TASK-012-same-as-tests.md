---
id: TASK-012
title: "QA: Test suite for SAME_AS embedding retrieval, multi-signal scoring, and LLM disambiguation"
story: STORY-005
assignee: qa-engineer
reporter: reza
priority: high
status: open
created: 2026-04-26
updated: 2026-04-26
blocked_by: [TASK-009, TASK-010, TASK-011]
blocks: []
wiki_refs: []
estimated: "2d"
branch: "agent/STORY-005/TASK-012-same-as-tests"
---

# TASK-012: STORY-005 Test Suite

## What

Write and run all tests for the SAME_AS semantic matching work. Begins when TASK-009,
TASK-010, and TASK-011 are all `in-review`.

### Unit tests:

1. **Multi-signal weight sum**: Assert `0.4 + 0.3 + 0.2 + 0.1 == 1.0` (sanity check;
   catches future weight drift).

2. **IBM alias resolution**: Create test entities `{name: "IBM", type: "Organization"}`
   and `{name: "International Business Machines", type: "Organization"}` with similar
   embeddings and identical context. Assert `final_score >= 0.85` and SAME_AS link created.

3. **Type disambiguation prevents incorrect merge**: Entities `{name: "Apple", type: "Organization"}`
   and `{name: "Apple", type: "Fruit"}` with high embedding similarity. Assert `type_compatibility = 0.0`
   brings `final_score < 0.85`; no SAME_AS link created.

4. **Partial name match (Jaro-Winkler)**: `"J. Smith"` vs `"John Smith"` — assert
   Jaro-Winkler score is > 0.80, pushing the candidate into the ambiguous zone [0.60, 0.85).

5. **Zero-neighbor context overlap**: Entity with no neighbors → `context_overlap = 0.0`,
   no ZeroDivisionError.

6. **Candidate threshold**: Vector candidates below 0.60 cosine similarity are not passed
   to the scorer.

7. **LLM disambiguation — YES path**: Mock LLM returning `DECISION: YES, CONFIDENCE: HIGH`.
   Assert SAME_AS link created with `method = "llm-disambiguated"`.

8. **LLM disambiguation — NO path**: Mock LLM returning `DECISION: NO`. Assert no SAME_AS
   link created.

### Integration test:

9. **End-to-end federation with alias**: Create graph A with entity `IBM (Organization)`.
   Create graph B with entity `International Business Machines (Organization)`. Run
   federation. Assert SAME_AS link exists between the two. Run federated query — assert
   entity appears once (deduplicated).

### Regression:

10. Run all existing federation tests; assert none regressed. Specifically verify that
    exact-match fast path still produces `confidence = 0.99` for identical names.

## Why

SAME_AS quality directly determines federation correctness. A false positive (merging
two different entities) is worse than a false negative (missing a link) — it corrupts
query results silently. These tests verify both directions.

## Scope

**In scope:**
- Unit tests 1-8
- Integration test 9
- Regression run 10

**Out of scope (explicit):**
- Performance/throughput testing (STORY-008)
- Cross-language entity matching

## Definition of Done

- [ ] All 10 test scenarios pass
- [ ] No existing federation tests regressed
- [ ] All STORY-005 acceptance criteria verified and marked complete in STORY-005
- [ ] Reviewed by: reza

## Output

- `test`: additions to federation service unit tests + integration tests

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

<!-- Not started. Blocked by TASK-009, TASK-010, TASK-011. -->
