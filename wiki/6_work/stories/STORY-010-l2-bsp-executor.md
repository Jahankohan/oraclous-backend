---
id: STORY-010
title: "L2 P0: BSP Executor — core agent execution loop (plan → execute → apply → terminate)"
type: architecture
layer: harnessing
reporter: reza
status: ready
priority: critical
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-2-harnessing-platform"]
tasks: []
decisions: ["ADR-006"]
---

# STORY-010: L2 BSP Executor (P0)

## Summary

The BSP (Bulk Synchronous Parallel / Pregel) Executor is the core agent execution
engine — the component nothing else in L2 works without. It runs the
plan → execute → apply → check-termination loop for every agent task. Each super-step
is one LLM interaction via the LLM Gateway. The Executor reads from AgentTask nodes in
L1, writes results back to L1, and triggers the next super-step or terminates.
This is a greenfield build; no equivalent exists in the current codebase.

## Problem Statement

- No agent execution engine exists in `oraclous-data-studio/`
- LangChain-based `llm_service.py` makes direct LLM calls but is not an execution loop
- No BSP loop, no super-step concept, no AgentTask lifecycle management
- L3 FTOps agents cannot run until the Executor exists and is tested

## Goals

- [ ] Implement BSP loop: read AgentTask from L1, call LLM Gateway (plan phase), dispatch tools (execute phase), write results to L1 (apply phase), check termination condition
- [ ] Implement AgentTask lifecycle: `created → in_progress → awaiting → completed/failed/cancelled`
- [ ] Implement ACP-compliant agent lifecycle API: 7 states, 3 execution modes (sync, async, stream)
- [ ] Implement crash recovery: on restart, scan L1 for `in_progress` AgentRun nodes older than heartbeat timeout → resume or fail
- [ ] Write health check endpoint: BSP Executor healthy = `200`, degraded (no LLM Gateway / no audit sidecar) = `503`
- [ ] Gate: Executor checks audit sidecar health at startup; refuses tasks if sidecar is unreachable

## Non-Goals

- Sub-agent delegation (`ctx.delegate`) — P2; explicit L3 gate (L3 cannot begin until P2 is implemented)
- Memory Store read/write (separate P1 story)
- HITL gate integration (P1 — separate story)

## Acceptance Criteria

- [ ] An AgentTask node created in L1 is picked up by the Executor within 5s
- [ ] A 3-super-step agent task (plan → 2 tool calls → answer) completes end-to-end against a real LLM Gateway
- [ ] AgentRun node in L1 transitions through `created → in_progress → completed` with timestamps at each transition
- [ ] Crashed Executor on restart resumes `in_progress` tasks or marks them `failed` after timeout
- [ ] Executor refuses new tasks if audit sidecar health check fails
- [ ] Integration test: task submitted via ACP sync API returns result within timeout

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Super-step timeout: global config or per-agent-type config? | engineering | open |
| 2 | ACP streaming: SSE or WebSocket? | engineering | open — SSE preferred (simpler) |

## Context & Background

- Full technical spec: `wiki/1_architecture/layer-2-harnessing-platform.md` § BSP Executor
- ADR-006: L2 is the Oraclous Harnessing Platform (BSP model decision)
- Existing code to migrate away from: `llm_service.py` direct LangChain calls (keep for now; Executor calls LLM Gateway instead)
- No existing BSP Executor code exists; this is greenfield
- Estimated effort: 3-4 weeks (P0 MVP)
