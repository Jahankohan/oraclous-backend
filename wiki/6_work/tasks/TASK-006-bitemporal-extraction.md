---
id: TASK-006
title: "Update LLM extraction prompt to request temporal bounds from source text"
story: STORY-002
assignee: ai-agent-specialist
reporter: reza
priority: critical
status: in-progress
created: 2026-04-26
updated: 2026-04-26
blocked_by: [TASK-005]
blocks: [TASK-008]
wiki_refs: []
estimated: "1-2d"
branch: "agent/STORY-002/TASK-006-bitemporal-extraction"
---

# TASK-006: Temporal-Aware LLM Extraction Prompt

## What

Update the entity/relationship extraction prompt in `app/services/pipeline_service.py`
to explicitly ask the LLM for temporal bounds on extracted relationships.

**Current prompt** (locate in `pipeline_service.py`): Extracts entities and relationships
but does not ask for when relationships were valid in the real world.

**New prompt addition** (add to the relationship extraction section):
```
For each relationship, also determine:
- event_time: When did this relationship start? Use ISO-8601 date if mentioned in the text. If not specified, return null.
- event_time_end: When did this relationship end, if it is no longer active? If still active or unknown, return null.

Examples:
- "John was CEO of Acme from 2018 to 2022" → event_time: "2018-01-01", event_time_end: "2022-12-31"
- "Apple acquired Intel in March 2025" → event_time: "2025-03-01", event_time_end: null
- "John knows Mary" (no date) → event_time: null, event_time_end: null
```

**Handle LLM output:**
- Parse `event_time` and `event_time_end` from the LLM response
- If the LLM returns a year only (`"2023"`), convert to `"2023-01-01"` (start of year)
- If null: store `null` on the relationship node (do not substitute ingestion_time)
- Write these values via the Cypher write path from TASK-005

**Also update `ingestion_source`** when writing extracted entities: set to the document
filename or connector ID that provided the extraction context.

**Files to modify:**
- `app/services/pipeline_service.py` — update extraction prompt template, parse temporal fields from response

## Why

The schema from TASK-005 provides the fields; this task fills them with meaningful values
for newly ingested content. Temporal queries (TASK-007) become accurate only when
event_time reflects actual real-world time, not ingestion time.

## Scope

**In scope:**
- Updating the entity/relationship extraction prompt in `pipeline_service.py`
- Parsing `event_time` and `event_time_end` from LLM response
- Setting `ingestion_source` to the document or connector source identifier

**Out of scope (explicit):**
- Retroactively re-extracting temporal data from already-ingested documents
- Temporal extraction for database connector rows (connector knows timestamps natively — handled separately)
- Chat service changes (TASK-007)

## Definition of Done

- [ ] Updated prompt is present in `pipeline_service.py` with temporal bound instructions and examples
- [ ] LLM response parser extracts `event_time` and `event_time_end` and writes them to Neo4j
- [ ] Test document with explicit date ("appointed CEO in 2020") produces relationship with `event_time = 2020-01-01`
- [ ] Test document without dates produces relationship with `event_time = null` (not ingestion_time)
- [ ] `ingestion_source` is set for all newly extracted entities
- [ ] Reviewed by: qa-engineer (TASK-008)

## Output

- `code`: `app/services/pipeline_service.py`

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

### 2026-04-26 — ai-agent-specialist — open → in-progress

Starting implementation. Branch: `agent/STORY-002/TASK-006-bitemporal-extraction`.
Updating LLM extraction prompt in pipeline_service.py to request temporal bounds from source text.
