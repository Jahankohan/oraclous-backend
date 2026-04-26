---
id: ADR-008
title: "Three valid deployment models with progressive onramp; layers are product boundaries, not standalone runtimes"
date: 2026-04-25
status: draft
story: ""
supersedes: ""
superseded_by: ""
layer: cross-cutting
authors: [cto, solution-architect]
---

# ADR-008: Three Valid Deployment Models

## Status

`draft` — 2026-04-25 · awaiting Reza acceptance

## Context

The SA raised that the architecture implies layers are "independently deployable" but
Layer 2 requires Layer 1 and Layer 3 requires Layers 1 and 2 — making the claim false
for L2 and L3 in isolation. The CTO found that the architecture documents describe three
product lines with explicit "Selling Layer X Alone" sections, framing the layers as
independent *product boundaries*, not independent *runtimes*.

## Decision

The three valid Oraclous deployment models are:

1. **Integration Fabric** (Layer 1 + MCP): A production-grade, multi-tenant knowledge
   graph with MCP-exposed ingestion and graph reasoning. For customers with their own
   agent framework who need the graph substrate.

2. **Harnessing + Graph** (Layers 1 + 2): The knowledge graph plus the Oraclous
   Harnessing Platform. For customers who want to run custom agent teams on a
   graph-native runtime without the full FTOps loop. Customers supply their own
   agent team; Oraclous supplies the runtime and graph substrate.

3. **End-to-End FTOps** (Layers 1 + 2 + 3): The full fine-tuning loop. The stock
   Oraclous agent team runs on the Harnessing Platform, on the graph.

Layer dependencies are: L2 requires L1 (the Harnessing Platform runs on the graph).
L3 requires L1+L2 (FTOps agents run on the Harnessing Platform). **No layer is
independently deployable without its dependencies.**

The phrase "independently deployable" shall not be used to describe the Oraclous layers.
The correct phrase is "independent product boundaries with progressive onramp."

## Rationale

- The three product lines serve three different customer maturity levels and budget points.
  This is a deliberate go-to-market strategy, not an architectural compromise.
- Framing the layers as independent products (rather than standalone runtimes) is accurate
  and commercially coherent. Layer 2 is not a product without Layer 1, but it IS a
  distinct product offering that does not require Layer 3.
- Removing "independently deployable" from the architecture description eliminates a
  false claim that will mislead future engineers and customers.

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| Truly independent runtimes (each layer runs without the others) | Would require L2 to duplicate L1's storage, L3 to duplicate L2's framework. Unnecessary duplication and the loss of graph-native properties. |
| Single monolithic product only | Reduces addressable market; customers who only need the graph substrate or the framework would be overcharged. |

## Consequences

**Positive:**
- Clear product strategy that matches the architecture's actual dependency structure.
- Sales and engineering share the same mental model.

**Negative / Trade-offs:**
- L2 and L3 cannot be evaluated independently of L1. Demos, POCs, and trials always
  include L1 as a prerequisite.

**Neutral:**
- The Integration Fabric is the lightest on-ramp; it is also the entry point for
  converting ecosystem tool-users into platform customers.

## Validation

If a customer can run Layer 2 without Layer 1 being deployed, this ADR has been violated.
The integration test should verify that L2's graph primitives fail gracefully without
a backing L1 instance.

## Related

- ADR-006 — L2 is framework-not-runtime
- ADR-007 — MCP-First (Integration Fabric product depends on this)
- Concern INDEPENDENT-LAYERS-CLAIM (closed by this ADR)
