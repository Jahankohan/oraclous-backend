---
id: STORY-002
title: "Implement true bitemporal tracking: separate event_time from ingestion_time"
type: feature
layer: knowledge-graph
reporter: reza
status: ready
priority: critical
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-1-knowledge-graph"]
tasks: [TASK-005, TASK-006, TASK-007, TASK-008]
decisions: []
---

# STORY-002: True Bitemporal Tracking

## Summary

The current temporal model conflates two independent timelines: `valid_from` is set to
ingestion time, not the time the fact was actually true in the world. This makes
temporal queries ("Who was CEO in 2023?") unreliable — we can't distinguish "a fact
that was true in 2023" from "a fact we ingested in 2023." Zep/Graphiti's bitemporal
model (proven with 18.5% accuracy improvement on DMR benchmark) tracks both independently.
This story adds `event_time`/`event_time_end` alongside existing `ingestion_time`.

## Problem Statement

- `valid_from` in `memory_service.py` is set to ingestion time, not world time
- Point-in-time queries ("what was true on date X?") return incorrect results
- Knowledge-as-of queries ("what did we know before Tuesday?") are not supported
- Ebbinghaus decay and temporal validity are orthogonal; both are needed but only decay exists

## Goals

- [ ] Add `event_time`, `event_time_end`, `ingestion_time`, `ingestion_source` to all entity and relationship nodes
- [ ] Migrate existing nodes: `event_time = ingested_at`, `ingestion_time = ingested_at` (backward-compatible)
- [ ] Update extraction prompt in `pipeline_service.py` to request temporal bounds from LLM
- [ ] Add three temporal query modes to chat: point-in-time, knowledge-as-of, changes-since
- [ ] Keep Ebbinghaus decay for agent memory (orthogonal, both run simultaneously)

## Non-Goals

- Changing the Ebbinghaus decay formula or memory consolidation logic
- Retroactively improving event_time accuracy for already-ingested documents (migration sets event_time = ingested_at)
- UI changes (temporal query mode is a backend API parameter)

## Acceptance Criteria

- [ ] Every newly ingested entity has `event_time`, `event_time_end`, `ingestion_time`, `ingestion_source` properties
- [ ] Existing entities get `event_time = ingested_at` via migration (no data loss)
- [ ] Point-in-time chat query `{"temporal_mode": "point_in_time", "at": "2023-12-31"}` returns only facts valid at that date
- [ ] Knowledge-as-of query `{"temporal_mode": "knowledge_as_of", "before": "2026-04-08"}` respects ingestion timeline
- [ ] LLM extraction prompt yields temporal bounds when source text contains dates (verified on 10 test documents)
- [ ] Ebbinghaus decay continues working correctly after migration

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Should event_time be nullable for facts without temporal bounds in source text? | engineering | open — yes, nullable |

## Context & Background

- Full technical spec: `ORACLOUS_DEEPENING_ROADMAP.md` § 6 (Bitemporal, pp. 356-438)
- Current impl: `app/services/memory_service.py` (lines 40-86, 146, 183), `app/services/pipeline_service.py` (lines 99-102)
- Cypher property additions: `app/models/`, `app/schemas/chat_schemas.py`
- Competitor: Zep/Graphiti — 18.5% improvement on Deep Memory Retrieval benchmark
- Estimated effort: 1 week
