# Stories — Requirements

Stories are requirements authored by Reza. The CTO decomposes each story into tasks via `/wiki split`.

---

## Index

| ID | Title | Layer | Status | Priority | Created |
|----|-------|-------|--------|----------|---------|
| [STORY-001](STORY-001-leiden-communities.md) | Leiden hierarchical communities (replace Louvain) | knowledge-graph | ready | critical | 2026-04-26 |
| [STORY-002](STORY-002-bitemporal-tracking.md) | True bitemporal tracking: event_time vs ingestion_time | knowledge-graph | ready | critical | 2026-04-26 |
| [STORY-003](STORY-003-relational-to-graph.md) | Relational-to-graph inference: schema mapper + row transformer | knowledge-graph | ready | high | 2026-04-26 |
| [STORY-004](STORY-004-code-kg-data-flow.md) | Code KG data flow analysis: FLOWS_TO edges + taint tracking | knowledge-graph | ready | medium | 2026-04-26 |
| [STORY-005](STORY-005-federation-same-as.md) | Federation SAME_AS semantic matching (replace exact match) | knowledge-graph | ready | high | 2026-04-26 |
| [STORY-006](STORY-006-multimodal-depth.md) | Multimodal depth: OCR fallback, diagram understanding, CSV/JSON/MD | knowledge-graph | ready | medium | 2026-04-26 |
| [STORY-007](STORY-007-frontend-mvp.md) | Frontend MVP: 4 screens with full backend integration | cross-cutting | ready | critical | 2026-04-26 |
| [STORY-008](STORY-008-benchmarks.md) | Benchmarks: ingestion, RAGAS quality, chat latency, federation overhead | cross-cutting | ready | high | 2026-04-26 |
| [STORY-009](STORY-009-production-hardening.md) | Production hardening: caching, pool metrics, degradation, rate limits | knowledge-graph | ready | high | 2026-04-26 |
| [STORY-010](STORY-010-l2-bsp-executor.md) | L2 P0: BSP Executor — core agent execution loop | harnessing | ready | critical | 2026-04-26 |
| [STORY-011](STORY-011-l2-llm-gateway.md) | L2 P0: LLM Gateway service | harnessing | ready | critical | 2026-04-26 |
| [STORY-012](STORY-012-l2-tool-dispatcher-scope-enforcer.md) | L2 P0: Tool Dispatcher + Scope Enforcer | harnessing | ready | critical | 2026-04-26 |
| [STORY-013](STORY-013-l2-coordination-layer.md) | L2 P0: Coordination Layer (APOC + AgentTask + Redis relay) | harnessing | ready | critical | 2026-04-26 |
| [STORY-014](STORY-014-l2-audit-sidecar.md) | L2 P0: Audit Emitter Sidecar | harnessing | ready | critical | 2026-04-26 |
| [STORY-015](STORY-015-l2-mcp-server.md) | L2 P0: MCP Server — extend to L2 capabilities | harnessing | ready | critical | 2026-04-26 |
| [STORY-016](STORY-016-l2-checkpointer.md) | L2 P0: Neo4j Checkpointer | harnessing | ready | critical | 2026-04-26 |
| [STORY-017](STORY-017-l2-hitl-gate.md) | L2 P1: HITL Gate Engine | harnessing | ready | high | 2026-04-26 |

---

## Execution Sequencing

### Track A — L1 Deepening (run in parallel with Track B)

```
STORY-001 (Leiden)   ←  critical, unblocked
STORY-002 (Bitemporal) ← critical, unblocked
STORY-005 (SAME_AS)  ← high, unblocked
STORY-009 (Hardening) ← high, unblocked
STORY-003 (Relational) ← high, after STORY-009 not required
STORY-007 (Frontend) ← critical, unblocked (separate team)
STORY-008 (Benchmarks) ← after STORY-001 + STORY-002 complete
STORY-004 (Code KG)  ← medium, unblocked
STORY-006 (Multimodal) ← medium, unblocked
```

### Track B — L2 Harnessing Platform (run in parallel with Track A)

```
P0 (all must complete before L3 can start):
  STORY-011 (LLM Gateway)      ← first; BSP Executor needs it
  STORY-014 (Audit Sidecar)    ← first; BSP Executor startup gate
  STORY-016 (Checkpointer)     ← first; BSP Executor crash recovery
  STORY-010 (BSP Executor)     ← after 011 + 014 + 016
  STORY-012 (Tool Dispatcher)  ← parallel with BSP Executor
  STORY-013 (Coordination)     ← parallel with BSP Executor
  STORY-015 (MCP Server)       ← after BSP Executor (P0 coequal per ADR-007)

P1 (after P0 complete):
  STORY-017 (HITL Gate)        ← requires STORY-016 + STORY-015
```

### Gate

**L3 FTOps agents cannot start until all L2 P0 stories are `done`.**
The explicit gate is `ctx.delegate` (P2) — L3 hierarchical agent work also requires P2.

---

## Story lifecycle

```
draft       ← Reza creates the story; details incomplete
ready       ← All acceptance criteria defined; open questions resolved; ready for CTO to split
in-progress ← CTO has split into tasks; at least one task is active
done        ← All tasks done; all acceptance criteria verified by QA
cancelled   ← Explicitly cancelled (keep for history)
```

## Template

See [_TEMPLATE.md](_TEMPLATE.md)
