---
id: ADR-009
title: "Layer 2 provides integration-layer capabilities; Layer 3 owns FTOps lifecycle stages"
date: 2026-04-25
status: draft
story: ""
supersedes: ""
superseded_by: ""
layer: harnessing
authors: [cto, solution-architect]
---

# ADR-009: L2 Capabilities vs. L3 Lifecycle Ownership

## Status

`draft` — 2026-04-25 · awaiting Reza acceptance

## Context

The SA raised that "evaluation" appears in both L2's responsibility list and L3's
fine-tuning lifecycle — creating overlap. The CTO found that the FTOps agent team
documents assign Evaluation Agents explicitly to Stage 7 of the L3 loop, while L2
exposes evaluation as an integration-layer capability (graph-grounded test set tooling,
metrics APIs, behavioral regression utilities that any agent or platform adapter can call).

## Decision

**Layer 2 provides integration-layer capabilities.** These are capabilities exposed via
MCP tools and platform adapters that any agent can invoke through their platform of choice:
graph traversal, analysis, prediction, write operations, evaluation metrics, monitoring
APIs, observability queries, feedback write operations. Layer 2 does not own any
lifecycle stage; it provides the capability surface.

**Layer 3 owns FTOps lifecycle stages.** The Evaluation Agents (Stage 7), Training Agents
(Stage 6), Monitor Agents (Stage 9), and all other FTOps agents are Layer 3 entities.
They consume L2 capabilities (via MCP or platform adapter) to do their work.

The word "evaluation" in any description of Layer 2 shall be understood to mean
"evaluation capability / MCP tool" — not "the evaluation lifecycle stage." Similarly,
"monitoring" in L2 means "monitoring APIs and MCP tools," not "the Monitor Agent."

## Rationale

- The framework/lifecycle distinction follows directly from ADR-006 (L2 is framework,
  not runtime). A framework provides primitives; a team owns lifecycle stages.
- Without this distinction, every L2 function would imply a competing L3 agent. The
  distinction eliminates the implied competition.
- Declarative agent skill files (`.oraclous.md`) specify which L2 capabilities each L3
  agent uses — the boundary is enforced at the skill file and platform adapter level.

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| Move evaluation entirely to L3, remove from L2 description | L2 exposes evaluation capabilities (RAGAS integration, test set generation) as MCP tools usable by any agent. Removing them from L2 would require L3 to duplicate tooling that belongs in the platform. |
| Move evaluation entirely to L2 (L3 calls L2 for evaluation) | This is what the architecture already says — L3 Evaluation Agents call L2 evaluation capabilities via MCP or adapter. The distinction is naming, not structure. |

## Consequences

**Positive:**
- No overlap between L2 and L3 when each layer is described with precise language.
- Custom AI Orchestration customers can use L2's evaluation primitives for their own
  agent teams without buying L3.

**Negative / Trade-offs:**
- Architecture documentation must be consistent in using "capabilities / MCP tools" for
  L2 and "lifecycle stages" for L3. Sloppy wording recreates the confusion.

**Neutral:**
- This ADR applies retroactively to all future documentation: bullet points under L2
  must say "evaluation capability / MCP tool," not "evaluation."

## Validation

If a future document describes Layer 2 as "owning" an FTOps lifecycle stage (not just
providing tools for it), this ADR has been violated.

## Related

- ADR-006 — L2 is framework-not-runtime
- Concern L2-L3-EVALUATION-OVERLAP (closed by this ADR)
