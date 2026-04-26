# Tasks — Execution Units

Tasks are created by the CTO when splitting stories. Each task has exactly one assignee.

---

## Index

| ID | Title | Assignee | Story | Status | Priority |
|----|-------|----------|-------|--------|----------|
| [TASK-001](TASK-001-leiden-algorithm-switch.md) | Switch community detection to leidenalg + hierarchy | backend-developer | STORY-001 | open | critical |
| [TASK-002](TASK-002-leiden-summaries.md) | Generate per-level LLM summaries for Leiden | ai-agent-specialist | STORY-001 | open | critical |
| [TASK-003](TASK-003-leiden-retriever.md) | Add COMMUNITY_SUMMARY retriever + global query routing | backend-developer | STORY-001 | open | critical |
| [TASK-004](TASK-004-leiden-tests.md) | QA: Leiden hierarchy, summaries, retriever tests | qa-engineer | STORY-001 | open | critical |
| [TASK-005](TASK-005-bitemporal-models.md) | Add bitemporal properties + migration | backend-developer | STORY-002 | open | critical |
| [TASK-006](TASK-006-bitemporal-extraction.md) | Update LLM extraction prompt for temporal bounds | ai-agent-specialist | STORY-002 | open | critical |
| [TASK-007](TASK-007-bitemporal-query-modes.md) | Add temporal query modes to chat service | backend-developer | STORY-002 | open | critical |
| [TASK-008](TASK-008-bitemporal-tests.md) | QA: Bitemporal properties, extraction, query modes | qa-engineer | STORY-002 | open | critical |
| [TASK-009](TASK-009-same-as-embeddings.md) | Embedding-based SAME_AS candidate retrieval | backend-developer | STORY-005 | open | high |
| [TASK-010](TASK-010-same-as-scorer.md) | Multi-signal SAME_AS scorer + EntityResolver | backend-developer | STORY-005 | open | high |
| [TASK-011](TASK-011-same-as-llm-disambiguation.md) | LLM disambiguation for ambiguous SAME_AS candidates | ai-agent-specialist | STORY-005 | open | high |
| [TASK-012](TASK-012-same-as-tests.md) | QA: SAME_AS retrieval, scoring, disambiguation | qa-engineer | STORY-005 | open | high |

---

## Dependency graph

```
STORY-001 (Leiden):
  TASK-001 → TASK-002 → TASK-003 → TASK-004
              ↑ (both block TASK-003)

STORY-002 (Bitemporal):
  TASK-005 → TASK-006 → TASK-008
  TASK-005 → TASK-007 → TASK-008

STORY-005 (SAME_AS):
  TASK-009 → TASK-010 → TASK-011 → TASK-012
```

All three stories are independent of each other and can run in parallel on separate branches.

---

## Task lifecycle

```
open        ← Created; not yet started
in-progress ← Assignee is actively working
blocked     ← Blocked by another task or external dependency
in-review   ← Work complete; PR open; waiting for QA or Security Architect sign-off
done        ← All DoD criteria met; PR merged; QA signed off
```

## Assignment rules

- One task = one assignee (no joint ownership)
- If work needs multiple agents → split into multiple tasks with proper `blocked_by` dependencies
- The CTO sets initial assignments; Reza can override

## Git workflow

Each task has a `branch:` field in its frontmatter. The assigned agent must:
1. Branch from `develop` using that exact branch name
2. Never push directly to `develop` or `main`
3. Open a PR when moving to `in-review`

See [agent-git-workflow.md](../5_research/agent-git-workflow.md) for the full convention.

## Template

See [_TEMPLATE.md](_TEMPLATE.md)
