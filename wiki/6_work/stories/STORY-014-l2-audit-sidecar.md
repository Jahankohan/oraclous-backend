---
id: STORY-014
title: "L2 P0: Audit Emitter Sidecar — append-only audit log with hash-chaining"
type: architecture
layer: harnessing
reporter: reza
status: ready
priority: critical
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-2-harnessing-platform"]
tasks: []
decisions: ["ADR-011"]
---

# STORY-014: L2 Audit Emitter Sidecar (P0)

## Summary

Every agent action must be auditable and tamper-evident. The Audit Emitter Sidecar
is a dedicated process that receives audit events from the BSP Executor, LLM Gateway,
and Scope Enforcer, and writes them as append-only `:AuditEvent` nodes to a dedicated
Neo4j partition. Events are hash-chained (each event's `hash` field includes the
previous event's hash) to detect tampering. The BSP Executor checks sidecar health
at startup and refuses to accept tasks if the sidecar is unreachable.

## Problem Statement

- No audit sidecar exists; agent actions are not recorded
- Without the sidecar, the Executor startup gate (STORY-010) has nothing to gate on
- Hash-chaining must be implemented in the sidecar (not in application code) to ensure
  the chain is maintained atomically

## Goals

- [ ] Build audit sidecar as a lightweight process (FastAPI or asyncio service) receiving events via internal endpoint
- [ ] Write `:AuditEvent` nodes to a dedicated Neo4j partition (separate database or label partition)
- [ ] Implement hash-chaining: each AuditEvent includes `hash = SHA256(prev_hash + event_data)`
- [ ] Expose health check endpoint: `GET /health` → `{status: "healthy" | "degraded"}`
- [ ] On sidecar failure: emit alert; BSP Executor pauses task acceptance; operator notification
- [ ] Sources that emit to sidecar: BSP Executor (task lifecycle), LLM Gateway (every LLM call), Scope Enforcer (every Cypher rewrite)

## Non-Goals

- Audit log query API (read access to audit log is via L1 directly for now)
- Audit log export or compliance reporting
- SOC2/EU AI Act certification (tracked separately in knowledge-base)

## Acceptance Criteria

- [ ] Every AgentRun state transition produces an AuditEvent node in L1 with correct `event_type`, `actor`, `graph_id`
- [ ] Hash chain is valid: `AuditEvent[n].hash == SHA256(AuditEvent[n-1].hash + AuditEvent[n].event_data)`
- [ ] Manual tampering with one AuditEvent node breaks chain (detectable by hash verification query)
- [ ] Sidecar health check returns `healthy` when Neo4j write is successful; `degraded` when Neo4j is unreachable
- [ ] BSP Executor starts normally when sidecar is healthy; refuses tasks when sidecar returns `degraded`

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Dedicated Neo4j database or label-based partition for audit events? | engineering | open |

## Context & Background

- Full technical spec: `wiki/1_architecture/layer-2-harnessing-platform.md` § Audit Emitter Sidecar
- ADR-011: Audit trail immutability — append-only partition, hash-chaining, dedicated writer
- AuditEvent schema: `{id, graph_id, event_type, actor, hash, prev_hash, created_at}`
- APOC `afterAsync` trigger approach for event emission from Neo4j → sidecar (spec §Audit)
- Estimated effort: 1 week
