---
id: ADR-005
title: "Layer 1 is the sole persistence layer; L2 memory writes to L1"
date: 2026-04-25
status: draft
story: ""
supersedes: ""
superseded_by: ""
layer: knowledge-graph
authors: [cto, solution-architect]
---

# ADR-005: Layer 1 Is the Sole Persistence Layer

## Status

`draft` — 2026-04-25 · awaiting Reza acceptance

## Context

Two SA concerns raised a related question: (1) L1 is described as "lean, single-purpose"
but also "supports agent coordination" — is coordination a second purpose? (2) L2 has
"memory" while L1 has a "knowledge base" — are these two separate stores?

These concerns resolve into one question: where does data live?

## Decision

Layer 1 (the knowledge graph) is the **only** layer where persistent data is stored.
Layer 2 does not maintain its own data store. L2's "memory" capability is a semantic
abstraction — it provides the interface and semantics for different memory types
(session, episodic, long-term) but all writes go to L1 graph nodes.

Coordination state (agent status, job queues, task assignments) is stored as graph nodes
in L1 — not because L1 has a "coordination" capability, but because Commitment 8 mandates
that every agent action is a graph write. Coordination emerges from graph writes; L1 does
not implement coordination logic.

## Rationale

- The architecture's auditability, replayability, and provenance guarantees all depend on
  a single authoritative store. Two stores means two sources of truth and two audit trails.
- Architectural Commitment 8 ("Every Agent Action Is a Graph Write") mandates this
  structurally — not by policy, but by design.
- L1 already provides versioning, bitemporal tracking, and ReBAC. L2's memory needs
  (session memory, episodic recall) benefit from all of these for free if stored in L1.

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| L2 maintains its own in-memory or Redis store | Creates two sources of truth. Breaks audit trail. Loses versioning, provenance, and ReBAC for agent memory. |
| Separate memory database (e.g., Zep, Mem0) | Same problem as above, plus vendor lock-in for memory specifically. |

## Consequences

**Positive:**
- Single source of truth. All audit, versioning, and provenance apply automatically to
  agent memory and coordination state.
- Commitment 8 is consistently enforced.

**Negative / Trade-offs:**
- Graph write latency applies to memory operations. High-frequency session state
  (sub-second) may require write batching or async writes. This is an implementation
  concern, not an architecture reversal.

**Neutral:**
- L1's description should be corrected from "lean, single-purpose" to "single-responsibility-
  for-persistence" to avoid future misreadings.

## Validation

A future implementation that introduces a second persistent store (Redis for memory,
separate log DB, etc.) violates this ADR and must either be justified by a superseding
ADR or reverted.

## Related

- Commitment 8 — Every Agent Action Is a Graph Write
- ADR-006 — L2 is framework-not-runtime
- Concern L1-SCOPE-CONTRADICTION (closed by this ADR)
- Concern L1-L2-MEMORY-BOUNDARY (closed by this ADR)
