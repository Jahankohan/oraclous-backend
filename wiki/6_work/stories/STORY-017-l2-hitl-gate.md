---
id: STORY-017
title: "L2 P1: HITL Gate Engine — human-in-the-loop checkpoint pause/resume with webhook notification"
type: architecture
layer: harnessing
reporter: reza
status: ready
priority: high
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-2-harnessing-platform"]
tasks: []
decisions: ["ADR-006"]
---

# STORY-017: L2 HITL Gate Engine (P1)

## Summary

The Human-in-the-Loop Gate Engine pauses agent execution at designated review points
and waits for a human operator to approve or reject a proposed action before proceeding.
Pause state is a `:HITLReview` node in L1 (persisted, survives restarts). A webhook
notifies the configured endpoint when a review is pending. A `POST /hitl/{review_id}/resolve`
REST endpoint (and MCP tool) allows operators to approve or reject. Default timeout is
24 hours; on timeout the task fails (no auto-approve).

## Problem Statement

- No HITL mechanism exists; agents execute all actions autonomously
- High-risk FTOps operations (model deployment, data deletion, fine-tuning config changes) need human review gates before L3 agents can be trusted with them
- Without HITL, L3 agents cannot be given write access to production artifacts

## Goals

- [ ] Implement `HITL gate` in BSP Executor: `await ctx.hitl.request_review(action, reviewer_role)` pauses the super-step
- [ ] Create `:HITLReview` node in L1 with `status=pending`, `timeout_at = now + 24h`, `action_proposed`, `reviewer_role`
- [ ] Send webhook notification to configured URL with review ID and action summary
- [ ] Implement REST endpoint: `POST /api/v1/hitl/{review_id}/resolve` with body `{approved: bool, reason: string}`
- [ ] On approval: resume BSP Executor from checkpoint; apply proposed action
- [ ] On rejection: mark AgentRun as `failed` with rejection reason
- [ ] On timeout (24h): Celery beat task marks review `timed_out`; AgentRun fails; no auto-approve
- [ ] MCP tool for HITL resolution (covered in STORY-015 — cross-reference)

## Non-Goals

- HITL UI (operators use REST endpoint or MCP tool; UI is a follow-up)
- Multi-reviewer workflows (single reviewer role per gate for MVP)
- Custom timeout per agent type (24h default is universal)

## Acceptance Criteria

- [ ] Agent calling `ctx.hitl.request_review(...)` causes AgentRun to enter `awaiting` state and a `:HITLReview` node appears in L1
- [ ] Webhook fires within 5s of review creation with correct `review_id` and `action_proposed`
- [ ] `POST /hitl/{review_id}/resolve` with `approved=true` → AgentRun resumes from checkpoint; action is applied
- [ ] `POST /hitl/{review_id}/resolve` with `approved=false` → AgentRun enters `failed` state; rejection reason stored on `:HITLReview`
- [ ] Review not resolved within 24h → Celery beat marks `timed_out`; AgentRun fails; verified by integration test
- [ ] Checkpoint from STORY-016 is used for resume (HITL depends on STORY-016 being done)

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Webhook retry policy: how many retries, what backoff? | engineering | open |
| 2 | Reviewer role enforcement: who is authorized to call the resolve endpoint? | reza | open |

## Context & Background

- Full technical spec: `wiki/1_architecture/layer-2-harnessing-platform.md` § HITL Gate Engine
- Depends on: STORY-016 (Checkpointer) — resume requires checkpoint; STORY-015 (MCP Server) — resolve tool
- HITLReview L1 schema in spec: `{id, run_id, graph_id, agent_type, action_proposed, reviewer_role, status, created_at, timeout_at, resolution}`
- SA concern HITL-PRIORITY-ORDERING (closed): HITL + REST resume endpoint must be coupled — this story implements both together
- Estimated effort: 1-2 weeks
