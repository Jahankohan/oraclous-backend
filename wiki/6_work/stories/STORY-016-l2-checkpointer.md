---
id: STORY-016
title: "L2 P0: Neo4j Checkpointer — BSP super-step state persistence with crash recovery"
type: architecture
layer: harnessing
reporter: reza
status: ready
priority: critical
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-2-harnessing-platform"]
tasks: []
decisions: ["ADR-005", "ADR-006"]
---

# STORY-016: L2 Neo4j Checkpointer (P0)

## Summary

The Checkpointer persists BSP super-step state to L1 after each super-step completes.
This enables crash recovery: if the BSP Executor dies mid-task, the Checkpointer data
allows resumption from the last completed super-step rather than re-running from the
beginning. `:Checkpoint` nodes live in L1 (ADR-005 compliance); reads and writes go
through the Scope Enforcer (tenant isolation); nodes have a TTL field cleaned up by
Celery beat. No external state store is used.

## Problem Statement

- BSP Executor state is in-memory only; process crash = task lost
- No `:Checkpoint` nodes exist in L1
- Without Checkpointer, crash recovery in STORY-010 would have to re-run from super-step 0
- LangGraph uses its own checkpointer (cannot be used; L2 is graph-native)

## Goals

- [ ] Implement `Neo4jCheckpointer` class: `save(run_id, super_step, state_json)` and `load(run_id)` methods
- [ ] All reads and writes go through the Scope Enforcer (graph_id enforced from JWT)
- [ ] `:Checkpoint` nodes include `ttl` property; Celery beat task cleans up expired checkpoints
- [ ] Checkpointer integrates with BSP Executor: checkpoint written after each super-step completes
- [ ] Crash recovery path in BSP Executor: on startup, scan for in-progress AgentRun nodes → load latest checkpoint → resume from that super-step

## Non-Goals

- External checkpointer (Redis, PostgreSQL) — L1 is the only persistence layer (ADR-005)
- Checkpoint history (only latest super-step checkpoint per run is needed)
- Checkpoint compression (JSON state is small enough for MVP)

## Acceptance Criteria

- [ ] BSP Executor writes a `:Checkpoint` node after each super-step; node has `run_id`, `graph_id`, `super_step`, `state_json`, `ttl`
- [ ] Kill BSP Executor after super-step 2 of a 5-super-step task; restart → task resumes from super-step 3 (not step 1)
- [ ] `:Checkpoint` nodes older than TTL are removed by Celery beat within 1 TTL cycle
- [ ] `Checkpointer.load(run_id)` with a non-existent run_id returns `None` (not an error)
- [ ] Checkpoint write fails if Scope Enforcer cannot enforce graph_id (task paused, not silently continued)

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | TTL value: 24h? 7d? Configurable per deployment? | engineering | open — propose 48h default |

## Context & Background

- Full technical spec: `wiki/1_architecture/layer-2-harnessing-platform.md` § Neo4j Checkpointer
- ADR-005: L1 is sole persistence — Checkpointer cannot use Redis or PostgreSQL as primary store
- Checkpoint node schema: `{run_id, graph_id, super_step, state_json, ttl, created_at}`
- Celery beat already exists in `oraclous-data-studio/background_jobs.py` — add TTL cleanup task there
- Estimated effort: 1 week
