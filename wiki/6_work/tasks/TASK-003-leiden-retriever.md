---
id: TASK-003
title: "Add COMMUNITY_SUMMARY retriever type and global query routing to chat service"
story: STORY-001
assignee: backend-developer
reporter: reza
priority: critical
status: open
created: 2026-04-26
updated: 2026-04-26
blocked_by: [TASK-001, TASK-002]
blocks: [TASK-004]
wiki_refs: ["layer-1-knowledge-graph"]
estimated: "2d"
branch: "agent/STORY-001/TASK-003-leiden-retriever"
---

# TASK-003: Community Summary Retriever and Global Query Routing

## What

Wire the Leiden community summaries (from TASK-002) into the retrieval path by adding
a new retriever type and auto-routing logic for global queries.

1. **Add `COMMUNITY_SUMMARY` to `app/schemas/retriever_schemas.py`**: Extend the
   `RetrieverType` enum with a new `COMMUNITY_SUMMARY` value.

2. **Implement `CommunitySummaryRetriever` in `app/services/retriever_factory.py`**:
   - Given a query string, select the appropriate hierarchy level (start with level-1
     or level-2 for broad questions)
   - Retrieve top-K `__Community__` nodes at that level (ranked by entity_count or
     relevance — use vector similarity on `summary` text if embeddings exist on communities,
     otherwise return top-N by size)
   - Return summaries as context for LLM answer generation

3. **Add global query auto-routing in `app/services/chat_service.py`**:
   - If `retriever_type` is not specified (auto mode), detect global vs specific:
     - **Global signals**: query contains no named entities AND contains words like
       "overview", "themes", "main topics", "across all", "summarize", "what are the"
     - **Specific signals**: query contains entity names, proper nouns, or specific IDs
   - Route global queries to `COMMUNITY_SUMMARY` retriever
   - Route specific queries to existing retrievers (vector, hybrid, text2cypher)

4. **Allow explicit override**: If the caller specifies `retriever_type=COMMUNITY_SUMMARY`
   explicitly, always use it regardless of query content.

**Files to modify:**
- `app/schemas/retriever_schemas.py` — add `COMMUNITY_SUMMARY` enum value
- `app/services/retriever_factory.py` — add `CommunitySummaryRetriever` class
- `app/services/chat_service.py` — add auto-routing logic

## Why

Without routing global queries to community summaries, thematic questions fall back to
flat vector search — which is the exact limitation MS GraphRAG's hierarchical community
summarization was designed to solve. This task closes that gap.

## Scope

**In scope:**
- New retriever type and implementation
- Keyword-heuristic auto-routing (entity-absence + keyword detection)
- Explicit `retriever_type` override support

**Out of scope (explicit):**
- LLM-based query classification (keyword heuristics are sufficient for MVP; LLM classification is a follow-up)
- Per-community embedding generation (use entity_count-based ranking if no embeddings on communities)
- Changing the existing retrievers

## Definition of Done

- [ ] `RetrieverType.COMMUNITY_SUMMARY` enum value exists and is documented in OpenAPI schema
- [ ] Chat request with `"what are the main themes?"` routes to `CommunitySummaryRetriever` (verified in test)
- [ ] Chat request with `"who is John Smith?"` routes to existing retriever (not community summary)
- [ ] Explicit `retriever_type=community_summary` always uses the community retriever
- [ ] Community summary answer references community-level insights, not raw entity property values
- [ ] Integration test: end-to-end global query returns a thematic summary
- [ ] Reviewed by: qa-engineer (TASK-004)

## Output

- `code`: `app/schemas/retriever_schemas.py`, `app/services/retriever_factory.py`, `app/services/chat_service.py`

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

<!-- Not started. Blocked by TASK-001, TASK-002. -->
