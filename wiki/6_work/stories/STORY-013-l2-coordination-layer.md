---
id: STORY-013
title: "L2 P0: Coordination Layer — APOC triggers, AgentTask lifecycle, Redis Streams relay"
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

# STORY-013: L2 Coordination Layer (P0)

## Summary

Agent wake-up and inter-agent coordination happen through graph writes. An APOC
`afterAsync` trigger fires when a new Finding/Entity/AgentTask node is created; this
trigger creates a downstream AgentTask node, which is the authoritative coordination
record (ADR-005). Redis Streams is the ephemeral delivery relay — it notifies the BSP
Executor that a new task exists, but the AgentTask node in L1 is what matters. A single
global trigger per event type (not per-tenant) avoids O(tenants × types) proliferation.

## Problem Statement

- No APOC triggers are installed; agent wake-up is not implemented
- No AgentTask lifecycle management exists
- No Redis Streams relay exists for task delivery
- Without coordination, agents cannot react to graph events or communicate through graph writes

## Goals

- [ ] Install global APOC `afterAsync` triggers per event type (not per-tenant) — single trigger filters `graph_id` internally
- [ ] Implement AgentTask node lifecycle: `pending → claimed → running → done/failed`
- [ ] Implement Redis Streams relay: APOC trigger creates AgentTask in L1 → relay publishes task ID to Redis stream → BSP Executor consumes stream, reads task from L1
- [ ] Implement BSP Executor task claiming with optimistic locking (prevent double-claim)
- [ ] Handle Redis Streams unavailability: BSP Executor polls L1 AgentTask nodes directly as fallback (above 500 tasks/hour, Redis is required — log warning below that threshold)

## Non-Goals

- Per-tenant APOC triggers (explicitly rejected — single global trigger per event type)
- Redis Streams as authoritative store (it is ephemeral relay only; AgentTask in L1 is authoritative per ADR-005)
- Dead letter queue (failed tasks stay in L1 with `failed` status; retry logic is in the Executor)

## Acceptance Criteria

- [ ] Creating a `:Finding {status: "new"}` node in L1 causes an `:AgentTask` node for `coverage-analyzer` to appear within 2s (APOC trigger fires)
- [ ] BSP Executor consumes Redis stream message, reads AgentTask from L1, claims it (status → `claimed`), begins execution
- [ ] A second Executor instance attempting to claim the same task fails gracefully (optimistic locking)
- [ ] Redis Streams unavailable: Executor falls back to L1 polling; tasks still execute (slower)
- [ ] No per-tenant triggers installed; one trigger per event type regardless of tenant count
- [ ] APOC trigger example from spec (`finding-created`) passes integration test

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Optimistic locking: CAS on `status` property or a `claimed_by` + `claimed_at` composite check? | engineering | open |

## Context & Background

- Full technical spec: `wiki/1_architecture/layer-2-harnessing-platform.md` § Coordination Layer
- ADR-005: L1 is sole persistence; Redis Streams is ephemeral relay — not a violation
- APOC trigger Cypher example in spec: single global `finding-created` trigger with internal `graph_id` filter
- Redis Streams already in docker-compose (used by Celery broker)
- Estimated effort: 2 weeks
