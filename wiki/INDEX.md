# Oraclous Wiki — Index

**This is the single source of truth for the Oraclous project.**

Maintained by the `/wiki` skill. Humans read it; the AI curator maintains it.
Last structure update: 2026-04-25

---

## Hierarchy of Truth

When content conflicts, this order governs — higher layers win:

```
0_foundation     ← immutable, highest authority
1_architecture   ← constrained by foundation
3_decisions      ← constrained by architecture
2_product        ← informed by all above
4_agents         ← informed by all above
5_research       ← reference only, no authority
6_work           ← must be consistent with all above
```

---

## Sections

### [0. Foundation](0_foundation/README.md)
The immutable core: mission, vision, founding principles, explicit non-goals.
**Volatility:** immutable — changes require Reza's explicit decision + a new ADR.

### [1. Architecture](1_architecture/README.md)
The three-layer architecture definition: Knowledge Graph, Harnessing Platform, FTOps.
Security, governance, and cross-cutting concerns.
**Volatility:** structural — changes only with major architectural pivots.

### [2. Product](2_product/README.md)
Competitive landscape, customer segments, GTM, roadmap.
**Volatility:** strategic — updated as market understanding evolves.

### [3. Decisions](3_decisions/README.md)
Architecture Decision Records (ADRs). Append-only, permanent record.
**Volatility:** append-only — past decisions are never deleted.

### [4. Agents](4_agents/_OVERVIEW.md)
Agent team definitions: roles, responsibilities, layer ownership, skills.
**Volatility:** semi-dynamic — evolves as capabilities and team structure change.

### [5. Research](5_research/README.md)
Background research, competitive analysis deep-dives, technical explorations.
**Volatility:** reference — append-only, no authority over other layers.

### [6. Work](6_work/stories/README.md)
Stories (requirements) and Tasks (execution units).
- [Stories →](6_work/stories/README.md)
- [Tasks →](6_work/tasks/README.md)

---

## Quick Commands

```
/wiki story "<title>"              Create a new requirement
/wiki task "<title>" --story XXX   Create a task under a story
/wiki split STORY-XXX              Decompose a story into tasks
/wiki adr "<title>"                Record an architecture decision
/wiki board                        Current work board (kanban view)
/wiki validate                     Check the wiki for conflicts
/wiki ingest <path>                Synthesize a document into wiki pages
/wiki query "<question>"           Answer a question from the wiki
```

---

## Key Concepts

- [[oraclous]] — what Oraclous is in one line
- [[three-layer-architecture]] — KG + Harnessing + FTOps
- [[ftops]] — Fine-Tuning DevOps, the category Oraclous defines
- [[knowledge-graph]] — Layer 1
- [[harnessing-platform]] — Layer 2: graph-native agent runtime (BSP, APOC coordination, LLM gateway, HITL, MCP, SDK)
- [[founding-principles]] — Open source, no vendor lock-in, data ownership, self-hosted
