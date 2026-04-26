---
id: STORY-009
title: "Production readiness: query caching, connection pool metrics, graceful degradation, rate limit refinement"
type: ops
layer: knowledge-graph
reporter: reza
status: ready
priority: high
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-1-knowledge-graph"]
tasks: []
decisions: []
---

# STORY-009: Production Readiness Hardening

## Summary

Several production gaps remain after the Phase 1-4 build: no query result caching
(every chat query hits Neo4j fresh), connection pool metrics are missing from OTEL
(so pool exhaustion is invisible), degradation paths for component failures are
unspecified (what happens when Neo4j or Redis are down?), and rate limits are flat
per-endpoint rather than per-tenant with burst allowance.

## Problem Statement

- P95 chat latency is ~3s for repeat queries that could be cached (targets: <1s for repeats)
- Connection pool exhaustion events are not visible in metrics (OTEL gaps)
- No documented or tested degradation path for Neo4j-down, Redis-down, Celery-down, LLM-down scenarios
- Rate limits are 30/minute flat; no per-tenant configuration; no burst allowance

## Goals

- [ ] Add Redis-based query result cache: key = hash(graph_id, query_text, retriever_type), TTL = 5 minutes, invalidate on ingest complete or entity update
- [ ] Add Neo4j connection pool metrics to OTEL: active connections, wait time, pool exhaustion events, Redis connection count
- [ ] Document and implement graceful degradation: 503 + retry-after (Neo4j down), bypass cache (Redis down), queue to PostgreSQL (Celery broker down), vector-only fallback (LLM down)
- [ ] Refine rate limiting: per-tenant configurable limits, burst allowance (token bucket), differential limits for read/write/admin endpoints
- [ ] Add structured error codes (KGB-XXXX format) to replace generic HTTP errors

## Non-Goals

- Neo4j clustering setup (infrastructure, not application code)
- Auto-scaling configuration
- Load testing tooling

## Acceptance Criteria

- [ ] Cache hit on repeated chat query: P95 latency drops from ~3s to <1s
- [ ] OTEL dashboard shows pool active/wait/exhaustion metrics (verified in Grafana or OTEL collector)
- [ ] Neo4j-down test: API returns 503 with `Retry-After` header; no unhandled exception in logs
- [ ] Redis-down test: chat query completes (slower, uncached); no 500 error
- [ ] Per-tenant rate limit config accepted via service account settings; burst of 10 requests allowed before sustained limit kicks in
- [ ] Error response for `graph_not_found` returns `{"error_code": "KGB-4001", "message": "...", "detail": "...", "docs_url": "..."}`

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | PostgreSQL queue fallback for Celery: use existing DB or add dedicated queue table? | engineering | open |

## Context & Background

- Full technical spec: `ORACLOUS_DEEPENING_ROADMAP.md` § 14 (Production Readiness, pp. 946-1010)
- OTEL setup exists; gap is connection pool instrumentation specifically
- Rate limiting: existing middleware in `app/middleware/` — extend, don't replace
- Estimated effort: 1-2 weeks
