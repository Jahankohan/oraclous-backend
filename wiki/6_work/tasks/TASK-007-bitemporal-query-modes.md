---
id: TASK-007
title: "Add three temporal query modes to chat service: point-in-time, knowledge-as-of, changes-since"
story: STORY-002
assignee: backend-developer
reporter: reza
priority: critical
status: in-progress
created: 2026-04-26
updated: 2026-04-26
blocked_by: [TASK-005]
blocks: [TASK-008]
wiki_refs: []
estimated: "2d"
branch: "agent/STORY-002/TASK-007-temporal-query-modes"
---

# TASK-007: Temporal Chat Query Modes

## What

Add an optional `temporal` parameter to the chat request that enables three query modes:

### Schema change (`app/schemas/chat_schemas.py`):
```python
class TemporalMode(str, Enum):
    POINT_IN_TIME = "point_in_time"    # facts valid at a specific real-world date
    KNOWLEDGE_AS_OF = "knowledge_as_of" # what the system knew before a given ingestion date
    CHANGES_SINCE = "changes_since"     # facts ingested after a given timestamp

class ChatRequest(BaseModel):
    # existing fields ...
    temporal_mode: Optional[TemporalMode] = None
    temporal_at: Optional[datetime] = None    # for point_in_time and knowledge_as_of
    temporal_since: Optional[datetime] = None  # for changes_since
```

### Retriever filter injection (`app/services/chat_service.py`):

When `temporal_mode` is set, inject a Cypher `WHERE` clause into the retriever query:

**point_in_time** (`temporal_at = T`):
```cypher
WHERE (r.event_time IS NULL OR r.event_time <= $T)
  AND (r.event_time_end IS NULL OR r.event_time_end >= $T)
```

**knowledge_as_of** (`temporal_at = T`):
```cypher
WHERE r.ingestion_time <= $T
```

**changes_since** (`temporal_since = T`):
```cypher
WHERE r.ingestion_time > $T
```

**Implementation note**: These filters apply to relationships. The simplest approach is
to add a `temporal_filter` optional parameter to the `RetrieverBase.retrieve()` method
and let each retriever that supports graph queries inject the filter into its Cypher.
Retrievers that use vector-only search can ignore the filter for now (log a warning).

### API backward compatibility:
- `temporal_mode` is optional; omitting it produces identical behavior to today
- Return a `temporal_mode_applied` field in the chat response confirming which filter was used (nullable)

**Files to modify:**
- `app/schemas/chat_schemas.py` — add `TemporalMode` enum and fields to `ChatRequest`
- `app/services/chat_service.py` — pass temporal filter to retriever
- `app/services/retriever_factory.py` — add `temporal_filter` optional param to retriever base

## Why

Without query modes, the temporal properties added by TASK-005 and TASK-006 are
invisible to callers. This task exposes them as a first-class API feature. It is
also the delivery mechanism that makes STORY-002's acceptance criteria verifiable.

## Scope

**In scope:**
- Three temporal query modes as optional chat request parameters
- Filter injection into Cypher-based retrievers
- `temporal_mode_applied` confirmation field in response
- Backward compatibility (no temporal mode = current behavior)

**Out of scope (explicit):**
- Vector-only retriever temporal filtering (log warning, continue without filter)
- UI changes
- Per-entity temporal filtering (filter applies to relationships only in MVP)

## Definition of Done

- [ ] `POST /api/v1/chat` accepts `temporal_mode`, `temporal_at`, `temporal_since` fields
- [ ] `point_in_time` query with `temporal_at = "2023-12-31"` returns only relationships where `event_time <= 2023-12-31 AND (event_time_end IS NULL OR event_time_end >= 2023-01-01)`
- [ ] `knowledge_as_of` query returns only facts with `ingestion_time <= temporal_at`
- [ ] `changes_since` query returns only facts with `ingestion_time > temporal_since`
- [ ] Omitting `temporal_mode` produces identical results to current behavior (regression test)
- [ ] OpenAPI schema updated with new fields and enum values
- [ ] Reviewed by: qa-engineer (TASK-008)

## Output

- `code`: `app/schemas/chat_schemas.py`, `app/services/chat_service.py`, `app/services/retriever_factory.py`

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

### 2026-04-26 — backend-developer — open → in-progress

Starting implementation. Branch: `agent/STORY-002/TASK-007-bitemporal-query-modes`.
Adding TemporalMode enum and temporal WHERE filter injection to chat_service.py and retriever_factory.py.
