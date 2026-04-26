# Agents

Agents are added incrementally as the architecture demands them.
Each agent is defined only when there is a specific, scoped function it needs to perform —
not because a role sounds useful.

## Current agents

| Slug | Role | Added | Skills |
|------|------|-------|--------|
| `solution-architect` | Evaluates architecture, detects conflicts, authors ADRs | 2026-04-25 | architectural-evaluation, security-governance-assessment, integration-surface-mapping, conflict-detection |
| `cto` | Holds institutional memory, retrieves evidence, responds to SA concerns, drafts ADRs | 2026-04-25 | institutional-memory, evidence-retrieval, gap-identification, adr-drafting |
| `qa-engineer` | Reviews completed tasks against DoD, writes test suites, signs off before `done` | 2026-04-26 | acceptance-criteria-verification, test-suite-authorship, regression-detection, integration-validation |
| `security-architect` | Reviews security-touching code changes, blocks on critical/high findings | 2026-04-26 | injection-detection, auth-review, credential-handling, tenant-isolation, input-validation, dependency-review |

See [definitions/solution-architect.md](definitions/solution-architect.md) · [definitions/cto.md](definitions/cto.md) · [definitions/qa-engineer.md](definitions/qa-engineer.md) · [definitions/security-architect.md](definitions/security-architect.md)

## How new agents are added

1. A concrete need is identified (a function nothing else handles)
2. Reza approves the new agent
3. Agent definition is written: role, skills, decision authority, interaction patterns, DoD
4. Skills are composed from existing skill files where possible; new skills written only if needed
5. Agent is added to this index
