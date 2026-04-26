---
id: TASK-XXX
title: ""
story: ""               # STORY-XXX parent (required unless standalone)
assignee: ""            # agent slug: cto | data-engineer | backend-developer |
                        # ai-agent-specialist | ai-integration-specialist |
                        # frontend-developer | security-architect |
                        # qa-engineer | devops-sre | tech-writer
reporter: reza
priority: critical | high | medium | low
status: open | in-progress | blocked | in-review | done
created: YYYY-MM-DD
updated: YYYY-MM-DD
blocked_by: []          # TASK-XXX ids this task cannot start before
blocks: []              # TASK-XXX ids that cannot start before this is done
wiki_refs: []           # [[concept]] pages the assignee must read before starting
estimated: ""           # rough effort: hours | days | 1w | 2w
---

# TASK-XXX: <Title>

## What

Concrete description of what needs to be done.
Specific enough that the assignee can start without asking clarifying questions.
If the assignee needs to make a design decision, state that explicitly here.

## Why

Context from the parent story. Why this task exists and what it enables.
One paragraph — the assignee needs to understand the goal, not just the steps.

## Scope

**In scope:**
- 

**Out of scope (explicit):**
- 

## Definition of Done

- [ ] Primary deliverable complete
- [ ] Tests written and passing (if code change)
- [ ] Relevant wiki pages updated
- [ ] If a decision was made: ADR created (ADR-XXX)
- [ ] Reviewed by: (agent slug or "reza")

## Output

What artifact this task produces:
- `code` — which files/services
- `doc` — which wiki page or external doc
- `decision` — which ADR
- `spec` — which design document
- `test` — which test suite

## Notes / Decisions Made

Running log of decisions made during execution. Add entries as they happen.

| Date | Decision | Rationale |
|------|----------|-----------|
|      |          |           |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

<!-- YYYY-MM-DD — agent-slug — status transition or action: what was done/decided/found -->
