---
id: ADR-007
title: "MCP-First: every capability exposed as both REST and MCP"
date: 2026-04-25
status: draft
story: ""
supersedes: ""
superseded_by: ""
layer: cross-cutting
authors: [cto, solution-architect]
---

# ADR-007: MCP-First Integration Protocol

## Status

`draft` — 2026-04-25 · awaiting Reza acceptance

## Context

The SA raised that the vision description omits MCP, leaving the integration surface
undefined and contradicting the "flexibility for integration" promise. The CTO found
that MCP is already Architectural Commitment 3 and the foundation of the Integration
Fabric product line, but this commitment has not been formalized as an ADR.

## Decision

Every capability Oraclous exposes is available as **both** a REST endpoint and an MCP
tool. No capability is MCP-only or REST-only.

MCP is the primary integration protocol for agent-to-agent and tool-to-tool communication.
It is not an optional adapter — it is the first-class interface. REST is the coequal
interface for human-facing tooling (dashboards, CLIs, scripts) and systems that predate MCP.

## Rationale

- MCP has become the de facto standard for tool-calling in LLM agent systems. Every
  new MCP server published in the ecosystem becomes immediately usable by Oraclous
  customers without any additional integration work.
- REST parity ensures backwards compatibility and supports non-agent callers.
- MCP-First is verifiable by customers: attempt to call any exposed capability via MCP;
  there should be no gaps.

## Alternatives Considered

| Alternative | Why Rejected |
|---|---|
| REST-only, with optional MCP adapters | Requires custom integration for every external agent; abandons the MCP ecosystem. |
| MCP-only | Breaks compatibility with existing HTTP tooling, monitoring systems, and non-MCP scripts. |
| gRPC or GraphQL as primary protocol | No ecosystem alignment for agent tool-calling. MCP specifically designed for this use case. |

## Consequences

**Positive:**
- Any MCP-compatible external agent (Claude Code, Cursor, Paperclip, any LangChain tool)
  can consume Oraclous capabilities without custom integration.
- Integration surface scales with the MCP ecosystem, not with Oraclous's connector count.

**Negative / Trade-offs:**
- Every new capability requires both a REST endpoint and an MCP tool definition.
  Doubles the surface area documentation requirement.
- MCP protocol evolution may require updates. Mitigation: MCP is now at Anthropic + broad
  ecosystem adoption; protocol stability is expected.

**Neutral:**
- The Integration Fabric product line is a direct consequence of this commitment and
  represents a viable standalone revenue line.

## Validation

MCP completeness is verifiable: call every REST endpoint via its MCP equivalent. If any
gap exists, this ADR has been violated. Testing should be automated in CI.

## Related

- Commitment 3 — MCP-First (this ADR formalizes it)
- ADR-006 — L2 is framework-not-runtime (external runtimes use L2 via MCP)
- Concern MCP-SURFACE-UNDEFINED (closed by this ADR)
