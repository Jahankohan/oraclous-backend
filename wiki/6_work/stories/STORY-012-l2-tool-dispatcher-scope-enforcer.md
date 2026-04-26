---
id: STORY-012
title: "L2 P0: Tool Dispatcher and Scope Enforcer — secure tool execution with Cypher rewrite"
type: architecture
layer: harnessing
reporter: reza
status: ready
priority: critical
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-2-harnessing-platform"]
tasks: []
decisions: ["ADR-006", "ADR-010"]
---

# STORY-012: L2 Tool Dispatcher + Scope Enforcer (P0)

## Summary

The Tool Dispatcher routes tool calls from the BSP Executor to the correct handler
(L1 graph operations, external APIs, sub-agent delegation). The Scope Enforcer is a
middleware layer that unconditionally rewrites every Cypher query to inject the
JWT-bound `graph_id` — agent-supplied graph_id values are overwritten, not just
validated. This is the security boundary that enforces tenant isolation on all agent
graph access. The Scope Enforcer is not optional; it intercepts every graph write.

## Problem Statement

- No tool dispatching infrastructure exists for agent tool calls
- `multi_tenant_components.py` has a known string-injection vulnerability (flagged by CTO)
- Current L1 API relies on well-behaved callers to supply correct `graph_id`; agents could forge it
- Scope inheritance (child agent tools are a subset of parent's grants) is not enforced

## Goals

- [ ] Implement Tool Dispatcher: tool call from BSP Executor → resolve handler → execute → return result
- [ ] Implement Scope Enforcer as Cypher rewrite middleware: strip agent-supplied graph_id from parameter bag, inject `$__enforced_graph_id` bound to JWT `graph_id` claim
- [ ] Fix string-injection vulnerability in `multi_tenant_components.py` (pre-existing issue)
- [ ] Implement scope inheritance check: reject tool calls wider than the service account's ACL grants
- [ ] Implement skill file ACL cross-validation at registration time: registration rejected if skill file reads/writes/tools are wider than service account ACL

## Non-Goals

- New tool types beyond graph operations and external REST calls (sub-agent delegation is P2)
- Tool result caching
- User-facing tool catalog API

## Acceptance Criteria

- [ ] Agent submits Cypher with `graph_id = "attacker-tenant"` → Scope Enforcer overwrites with JWT `graph_id`; attacker-tenant graph is never touched
- [ ] String-injection vulnerability in `multi_tenant_components.py` is fixed (parameterized, no format strings in Cypher)
- [ ] Skill file registering `WRITE` on graph X but service account has only `READ` → registration rejected with descriptive error
- [ ] Child agent tool call for graph Y (not in parent's grants) → rejected at Scope Enforcer, not at L1
- [ ] Tool dispatch round-trip for a graph read tool: <100ms overhead on local deployment

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Scope Enforcer placement: middleware in knowledge-graph-builder or in a separate sidecar? | engineering | open |

## Context & Background

- Full technical spec: `wiki/1_architecture/layer-2-harnessing-platform.md` § Tool Dispatcher, § Security Layer
- ADR-010: Scope inheritance and agent authorization
- CTO finding: string-injection in `multi_tenant_components.py` — fix this here
- Scope Enforcer Cypher rewrite example in spec: agent param `$gid` is overwritten, not just checked
- Estimated effort: 2 weeks
