---
id: ADR-010
title: "Scope inheritance is a platform-enforced architectural invariant; Phase 3 adds per-operation permissions"
date: 2026-04-25
status: draft
story: ""
supersedes: ""
superseded_by: ""
layer: cross-cutting
authors: [cto, solution-architect]
---

# ADR-010: Scope Inheritance and Agent Authorization

## Status

`draft` — 2026-04-25 · awaiting Reza acceptance

## Context

The SA raised that Layer 2 provides no structural mechanism to prevent a sub-agent from
exceeding its parent's scope (confused-deputy problem). The CTO found that Architectural
Commitment 7 (Scope Inheritance) and ADR-004 (Agent Service Accounts) address this
structurally, but per-operation permissions are planned for Phase 3.

## Decision

**Scope inheritance is a platform-enforced architectural invariant** (Commitment 7):
child agents' tools and access are enforced to be a subset of their parent's, checked
by the platform infrastructure — not trusted to agent skill files or application code.

**Agent Service Accounts** (ADR-004) provide per-agent scoped credentials. An agent that
does not have a graph grant cannot access a sub-graph, regardless of what its skill file
claims.

**Phase 3 delivery commitment:** Per-operation permissions — where each LLM request gets
a fresh permission check scoped to the specific query — shall be delivered in Phase 3.
Until Phase 3, the structural defense is scope inheritance + agent service account ACLs.
The confused-deputy threat is acknowledged and partially mitigated; full mitigation
requires Phase 3.

## Rationale

- Application-layer enforcement of scope (e.g., in agent skill files) can be bypassed by
  a compromised or malicious agent. Platform-level enforcement cannot.
- ADR-004 already establishes agent service accounts with admin-managed grants — this
  ADR formalizes the invariant and adds the Phase 3 commitment.
- Per-operation permissions (fresh check per LLM request) are the strongest defense
  against confused-deputy attacks. Sequencing them to Phase 3 is an explicit risk
  acceptance, recorded here.

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| Trust agent skill files for scope enforcement | Bypassable. The security threat model explicitly documents this and rejects it (ADR-004 rationale). |
| Full per-operation permissions from Day 1 | Implementation complexity would block earlier phases. The scope inheritance + service account ACLs provide meaningful structural defense in the interim. |

## Consequences

**Positive:**
- Blast radius of any compromised agent is bounded by its parent chain.
- Admin has full control over which agents access which sub-graphs (ADR-004).

**Negative / Trade-offs:**
- Until Phase 3, the confused-deputy attack vector is partially open. A high-privilege
  parent agent can still be used as a proxy for lower-privilege work. The SA's concern
  stands as a known risk until Phase 3 delivery.

**Risk acceptance:** The gap between current enforcement (scope inheritance + ACLs)
and full enforcement (per-operation permissions) is explicitly accepted until Phase 3.
Reza must confirm this acceptance.

**Neutral:**
- Microsoft Agent Governance Toolkit (April 2026) is under evaluation as an alternative
  to custom per-operation permission work. If adopted, it may accelerate Phase 3.

## Validation

- Spawn a child agent with a tool the parent lacks → platform rejects it.
- Attempt a cross-graph query without a graph grant → platform rejects it.
- Post-Phase 3: a single LLM request cannot perform operations outside the scope of
  the query that triggered it.

## Related

- ADR-004 — Agent sub-graph access via admin-managed ACL
- Commitment 7 — Scope Inheritance
- Concern AGENT-SCOPE-CONTAINMENT (partially closed; Phase 3 gap remains open)
