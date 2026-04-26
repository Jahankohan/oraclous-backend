# Architecture — Three-Layer Platform

**Volatility:** Structural. Changes require a new ADR and CTO + Reza approval.

**Authority:** Second highest. Constrained by `0_foundation/`. All product, agent, and work decisions are constrained by this section.

---

## The Three Layers

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 3: FTOps                                              │
│  End-to-end fine-tuning lifecycle as a team of agents        │
│  Runs on any L2-supported platform                           │
├──────────────────────────────────────────────────────────────┤
│  LAYER 2: Harnessing Platform                                │
│  Graph-native agent runtime: BSP executor, APOC-triggered    │
│  coordination, LLM gateway, HITL, MCP server+client,         │
│  Python+TS SDK, skill files. External platforms connect via  │
│  MCP (no custom adapters).                                   │
├──────────────────────────────────────────────────────────────┤
│  LAYER 1: Knowledge Graph                                    │
│  Multi-tenant data substrate, ReBAC, versioning, federation  │
│  Designed from the start for integration with any platform   │
└──────────────────────────────────────────────────────────────┘
```

Each layer has an independent product boundary with a progressive onramp (ADR-008).
Layers are not independently deployable runtimes — L2 requires L1, L3 requires L1+L2.

---

## Pages in this section

| Page | Contents | Status |
|------|----------|--------|
| [layer-1-knowledge-graph.md](layer-1-knowledge-graph.md) | KG design, principles, what exists, deepening work streams | created |
| [layer-2-harnessing-platform.md](layer-2-harnessing-platform.md) | Harnessing Platform: BSP runtime, coordination, LLM gateway, HITL, SDK, MCP | created |
| [layer-3-ftops.md](layer-3-ftops.md) | FTOps loop, 16 agents, feedback closure | _to be created_ |
| [security-governance.md](security-governance.md) | Cross-cutting security, audit, compliance | _to be created_ |
| [cross-cutting-principles.md](cross-cutting-principles.md) | MCP-first, provenance everywhere, bitemporal | _to be created_ |

---

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | L2 Harnessing Platform implementation — SA + CTO review complete; ADR-006 accepted | SA + CTO | closed |

---

## How to add a page here

Via `/wiki concept "<title>" --layer architecture`. The wiki skill will validate against `0_foundation/` before writing.
