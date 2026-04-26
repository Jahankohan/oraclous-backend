---
id: STORY-001
title: "Replace Louvain with Leiden; wire hierarchical community summaries into retrieval"
type: feature
layer: knowledge-graph
reporter: reza
status: ready
priority: critical
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-1-knowledge-graph"]
tasks: [TASK-001, TASK-002, TASK-003, TASK-004]
decisions: []
---

# STORY-001: Leiden Hierarchical Communities

## Summary

The community detection algorithm is currently GDS Louvain despite `leidenalg` being
present in `requirements.txt`. Leiden produces higher-quality communities (proven in the
original paper) and enables true hierarchical summarization — the core innovation of
MS GraphRAG's global search. Without hierarchy in the retrieval path, broad thematic
queries ("what are the main topics across all documents?") fall back to flat vector
search, defeating the purpose of community detection entirely.

## Problem Statement

- `analytics_service.py` calls Neo4j GDS Louvain; `leidenalg` is installed but unused
- Multi-level community hierarchy is stored but not used in the chat/retrieval path
- Global queries route to flat vector search, not community summaries
- MS GraphRAG's key differentiator (hierarchical global search) is unimplemented

## Goals

- [ ] Replace GDS Louvain call with `leidenalg.find_partition` on extracted igraph
- [ ] Implement multi-resolution Leiden (resolutions: 0.25, 0.5, 1.0, 2.0, 4.0) to produce parent-child community hierarchy
- [ ] Generate per-level LLM summaries (level-appropriate abstraction prompts)
- [ ] Add `COMMUNITY_SUMMARY` retriever type (6th type) to `retriever_factory.py`
- [ ] Add global query routing to `chat_service.py` (keyword heuristics + entity-absence detection)

## Non-Goals

- Changing the community node schema beyond adding `summary_level` and `PARENT_OF` edges
- Supporting directed graphs (Leiden runs on undirected entity graph)
- Per-tenant Leiden parameter tuning (single global config is sufficient for now)

## Acceptance Criteria

- [ ] `detect_communities_task` uses `leidenalg` not GDS Louvain; produces communities with `level` field and `PARENT_OF` edges between levels
- [ ] LLM summaries exist at each level with level-appropriate prompts (specific → abstract)
- [ ] Chat request with query "what are the main themes?" routes to community summary retriever, not vector retriever
- [ ] Community summary retriever answer references community-level insights, not individual entity properties
- [ ] Existing community detection tests pass; 3 new tests added (hierarchy structure, summary generation, global query routing)

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Resolution parameter values (0.25–4.0) or tune per graph size? | engineering | open |

## Context & Background

- Full technical spec: `ORACLOUS_DEEPENING_ROADMAP.md` § 5 (Leiden, pp. 258-353)
- Current impl: `app/services/analytics_service.py`, `app/services/background_jobs.py`
- Retriever factory: `app/services/retriever_factory.py`, `app/schemas/retriever_schemas.py`
- MS GraphRAG reference: hierarchical community summarization is their core claim
- Estimated effort: 1-2 weeks
