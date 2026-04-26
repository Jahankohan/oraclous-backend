---
id: ADR-006
title: "Layer 2 is the Oraclous Harnessing Platform — a graph-native agent runtime built for the L1 graph substrate"
date: 2026-04-25
status: draft
story: ""
supersedes: ""
superseded_by: ""
layer: harnessing
authors: [cto, solution-architect, reza]
---

# ADR-006: Layer 2 Is the Oraclous Harnessing Platform

## Status

`draft` — 2026-04-25 · awaiting Reza acceptance

## Context

The L1 graph substrate is the coordination medium, the shared memory, and the audit log.
An agent runtime that was not built for it cannot use those properties natively — APOC
triggers, bitemporal writes, graph-scoped credentials, and graph traversal for memory
are not features you bolt on to a general-purpose runtime.

External platforms (LangGraph, CrewAI, AutoGen, Paperclip, Claude Code) connect via
MCP (ADR-007). That is the full integration story. No custom adapters are built or planned.

## Decision

**Layer 2 is the Oraclous Harnessing Platform** — a graph-native agent runtime built
specifically for the L1 knowledge graph substrate.

It provides:
1. An agent execution engine (BSP loop, tool dispatch, LLM calls)
2. Graph-native coordination (APOC triggers → event relay → agent wake-up)
3. Inter-agent messaging through graph writes (the graph IS the message bus)
4. HITL gates via checkpoint pause/resume
5. Server-side scope enforcement at the L1 API boundary
6. LLM provider gateway (provider-agnostic, not provider-specific)
7. All integration surfaces: MCP server, MCP client, REST API, Python SDK, TypeScript SDK, skill files

External platforms connect via MCP (ADR-007). No custom platform adapters are built.
Any MCP-compatible runtime can consume L2 capabilities as tools. The Oraclous runtime
is the first-class path; MCP is the universal fallback.

**Layer name:** "Harnessing Platform" — the historical placeholder name is now the
correct name. It was always what this was going to be.

## Rationale

- A runtime built for a graph substrate can use APOC triggers, bitemporal writes, graph
  traversal for memory retrieval, and graph-scoped credential isolation natively. An
  adapter over LangGraph or CrewAI cannot — those runtimes use their own state stores
  and their own execution loops.
- The BSP (Bulk Synchronous Parallel) execution model used by production agent runtimes
  maps directly to a graph: nodes execute, write to channels, trigger downstream nodes.
  Building BSP over Neo4j is the same problem at a different storage layer.
- MCP as the external integration surface (ADR-007) means any external platform gets
  access to L2 capabilities without us building adapters for them. The integration
  requirement is fully met by MCP.
- Founding Principle 2 (No Vendor Lock-in) is better served by owning the runtime than
  by depending on external runtimes that can change APIs, change licensing, or discontinue.

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| L2 as a thin integration layer over LangGraph/CrewAI | Rejected by Reza: if the integrations are tricky, skip them. We need graph-native coordination, not adapter wrappers. |
| L2 as "framework not a runtime" | Rejected by Reza: competitive positioning is not an architecture argument. |
| MCP-only with no runtime | Leaves L3 FTOps agents without a first-class execution environment. Customers would need to supply a runtime to use L3 at all. |

## Consequences

**Positive:**
- L3 FTOps agents run in a runtime that natively understands the graph — bitemporal
  writes, scope inheritance, audit emission are built-in, not bolted on.
- No dependency on external frameworks that can break APIs or change licensing.
- The runtime can implement graph-native features (trigger-based agent wake-up, graph
  traversal for memory) that no general-purpose runtime provides.

**Negative / Trade-offs:**
- We build and maintain the runtime. This is engineering investment up-front.
- Customers who prefer LangGraph or CrewAI use MCP. They get full capability but not
  the ergonomic depth of the native runtime. This is acceptable: MCP is a complete
  integration surface.

## Validation

- An L3 FTOps agent runs end-to-end on the Oraclous runtime with zero external
  orchestration framework dependency.
- An agent write triggers a downstream agent within the APOC trigger → event relay path.
- A Claude Code session consuming L2 via MCP can invoke the same operations as a native
  Oraclous runtime agent and produces identical graph writes.

## Related

- ADR-005 — L1 is the sole persistence layer (runtime checkpointer writes to L1)
- ADR-007 — MCP-First (external integration surface)
- ADR-008 — Three deployment models
- ADR-010 — Scope inheritance
- ADR-011 — Audit trail
- [wiki/1_architecture/layer-2-harnessing-platform.md](../1_architecture/layer-2-harnessing-platform.md)
- Concern L2-HARNESSING-PLATFORM-BOUNDARY (closed)
