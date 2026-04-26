---
id: STORY-015
title: "L2 P0: MCP Server — extend to L2 capabilities, coequal with runtime from day one"
type: architecture
layer: harnessing
reporter: reza
status: ready
priority: critical
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-2-harnessing-platform"]
tasks: []
decisions: ["ADR-007"]
---

# STORY-015: L2 MCP Server (P0)

## Summary

An MCP server skeleton exists in `oraclous-data-studio/` covering L1 tools (15 tools).
ADR-007 (MCP-First) requires MCP to be coequal with the runtime from day one — not a
P1 afterthought. External runtimes (Claude Code, Paperclip, LangGraph) connecting via
MCP must have access to L2 capabilities (agent submission, task status, HITL resolution,
memory read/write) as MCP tools. This story extends the existing MCP server with L2
tools and fixes the L1 coverage gaps identified in the ADR-007 audit.

## Problem Statement

- Existing MCP server covers L1 only; no L2 tools exist
- Some L1 tools lack REST parity (ADR-007 violation — flagged in CTO evidence review)
- External runtimes have no way to submit agent tasks or check task status via MCP
- HITL resolution, memory access, and audit queries are not available as MCP tools

## Goals

- [ ] Add L2 MCP tools: `submit_agent_task`, `get_task_status`, `cancel_task`, `list_agent_runs`
- [ ] Add HITL MCP tool: `resolve_hitl_review` (approve/reject a pending HITLReview node)
- [ ] Add Memory MCP tools: `memory_get`, `memory_set`, `memory_list` (scoped to caller's graph_id from JWT)
- [ ] Fix L1 tool REST parity gaps from ADR-007 audit (enumerate and fix missing dual-surface tools)
- [ ] Ensure all MCP tools pass Scope Enforcer (graph_id from JWT, not from tool call parameters)
- [ ] MCP server runs in the same container as the L2 runtime (not a separate deployment)

## Non-Goals

- MCP client implementation (L2 as MCP client calling external tools) — P1 follow-up
- Python SDK (STORY-017) — separate story
- Admin MCP tools (tenant management, service account management)

## Acceptance Criteria

- [ ] Claude Code session can call `submit_agent_task` via MCP and receive a task ID
- [ ] `get_task_status` returns current AgentRun state for a submitted task
- [ ] `memory_set` followed by `memory_get` in the same Claude Code session returns the stored value
- [ ] `resolve_hitl_review` with `approved=true` updates the HITLReview node status and resumes the paused agent run
- [ ] All MCP tool calls are rejected if JWT `graph_id` is absent or invalid
- [ ] No L1 tool gap from ADR-007 audit remains unresolved

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | List the specific L1 tools lacking REST parity (need ADR-007 audit results) | engineering | open |

## Context & Background

- Full technical spec: `wiki/1_architecture/layer-2-harnessing-platform.md` § Integration Surfaces (MCP Server P0)
- ADR-007: MCP-First — every L2 capability must be available as an MCP tool
- Existing MCP skeleton: `oraclous-data-studio/` — locate the existing MCP server file
- Estimated effort: 1-2 weeks (L2 tools + L1 gap fixes)
