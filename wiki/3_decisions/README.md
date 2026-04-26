# Decisions — Architecture Decision Records

**Volatility:** Append-only. Past decisions are never deleted or modified. If a decision is superseded, the old ADR gets `status: superseded` + a reference to the new ADR.

**Authority:** Constrained by `0_foundation/` and `1_architecture/`. An ADR cannot override a founding principle.

---

## Index

| ID | Title | Status | Date | Layer |
|----|-------|--------|------|-------|
| [ADR-001](ADR-001-rebac-new-graphs-only.md) | ReBAC applied to new graphs only | accepted | — | knowledge-graph |
| [ADR-002](ADR-002-cross-graph-federation-opt-in.md) | Cross-graph federation is opt-in | accepted | — | knowledge-graph |
| [ADR-003](ADR-003-versioning-snapshot-approach.md) | Versioning via snapshot approach | accepted | — | knowledge-graph |
| [ADR-004](ADR-004-agent-subgraph-access-model.md) | Agent subgraph access model | accepted | — | knowledge-graph |
| [ADR-005](ADR-005-l1-sole-persistence-layer.md) | L1 is the sole persistence layer | draft | 2026-04-25 | cross-cutting |
| [ADR-006](ADR-006-l2-harnessing-platform.md) | Layer 2 is the Oraclous Harnessing Platform | draft | 2026-04-25 | harnessing |
| [ADR-007](ADR-007-mcp-first.md) | MCP-First integration protocol | draft | 2026-04-25 | cross-cutting |
| [ADR-008](ADR-008-deployment-models.md) | Three valid deployment models | draft | 2026-04-25 | cross-cutting |
| [ADR-009](ADR-009-l2-l3-primitive-vs-lifecycle.md) | L2 primitives vs L3 lifecycle ownership | draft | 2026-04-25 | harnessing |
| [ADR-010](ADR-010-scope-inheritance.md) | Scope inheritance and agent authorization | draft | 2026-04-25 | cross-cutting |
| [ADR-011](ADR-011-audit-trail-immutability.md) | Audit trail immutability mechanism | draft | 2026-04-25 | cross-cutting |
| [ADR-012](ADR-012-provider-abstraction.md) | LLM and training framework provider abstraction | superseded | 2026-04-25 | cross-cutting |
| [ADR-013](ADR-013-model-artifact-isolation.md) | Model artifact storage and tenant isolation | draft | 2026-04-25 | ftops |
| [ADR-014](ADR-014-llm-gateway.md) | LLM Gateway + training framework adapter (supersedes ADR-012) | draft | 2026-04-25 | harnessing |

_New ADRs are created via `/wiki adr "<title>"`_

---

## Template

See [_TEMPLATE.md](_TEMPLATE.md)

---

## When to write an ADR

Write an ADR when:
- A decision affects more than one layer
- A decision cannot easily be reversed
- A decision was made after considering and rejecting alternatives
- A decision contradicts what the team might expect (needs explanation)
- A decision involves a security or compliance tradeoff

Do NOT write an ADR for:
- Routine implementation choices (variable names, minor refactors)
- Decisions that are trivially reversible
- "We'll use X library" unless it has significant lock-in implications
