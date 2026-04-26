---
id: STORY-005
title: "Replace exact SAME_AS matching with embedding-based multi-signal entity resolution"
type: feature
layer: knowledge-graph
reporter: reza
status: ready
priority: high
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-1-knowledge-graph"]
tasks: [TASK-009, TASK-010, TASK-011, TASK-012]
decisions: []
---

# STORY-005: Federation SAME_AS Semantic Matching

## Summary

Cross-graph entity deduplication (SAME_AS) currently uses exact name+type matching,
producing confidence 0.99 for exact matches and missing everything else. The store
threshold (0.85) and candidate threshold (0.60) are defined but unused — they were
placeholders for embedding-based matching that was planned but never implemented.
Federation quality depends directly on SAME_AS link quality; the MVP implementation
misses aliases, partial names, and entity type disambiguation.

## Problem Statement

- "IBM" and "International Business Machines" are not linked (alias miss)
- "J. Smith" and "John Smith" are not linked (partial name miss)
- "Apple Inc." (Organization) vs "Apple" (no type) may be incorrectly merged (disambiguation failure)
- Thresholds (0.85/0.60) are defined in code but ignored; only exact match runs
- Federation deduplication is effectively broken for real-world entity names

## Goals

- [ ] Implement embedding-based candidate retrieval: vector search over target graph entities using existing embeddings on `__Entity__` nodes
- [ ] Implement multi-signal scorer: embedding similarity (0.4) + name similarity via Jaro-Winkler (0.3) + type compatibility (0.2) + neighbor context Jaccard (0.1)
- [ ] Apply defined thresholds: candidates above 0.60, store SAME_AS above 0.85
- [ ] Add LLM-assisted disambiguation for scores in [0.60, 0.85] range
- [ ] Centralize resolution logic in `app/components/entity_resolver.py` (used by federation and intra-graph dedup)

## Non-Goals

- Retroactive re-resolution of existing SAME_AS links (forward-only; re-run federation to update)
- Changing SAME_AS storage format (MERGE idempotency preserved)
- Cross-language entity matching (e.g., "IBM" in English vs "IBM" in Japanese)

## Acceptance Criteria

- [ ] "IBM" resolves as SAME_AS to "International Business Machines" (same graph, two entities)
- [ ] "Apple Inc." (type: Organization) does NOT match "Apple" (type: Fruit) — type weight prevents it
- [ ] "J. Smith" resolves as SAME_AS to "John Smith" with confidence ≥ 0.80
- [ ] LLM disambiguation called only for scores in [0.60, 0.85) range, not for clear matches
- [ ] Multi-signal score calculation is unit-tested with correct weights summing to 1.0
- [ ] Federated query returns deduplicated results for a test dataset with known aliases

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Neighbor context window: 1-hop or 2-hop for Jaccard? | engineering | open |

## Context & Background

- Full technical spec: `ORACLOUS_DEEPENING_ROADMAP.md` § 9 (Federation SAME_AS, pp. 609-674)
- Current impl: `app/services/federation_service.py` (lines 353-376)
- `__Entity__` nodes already have embeddings — use these for vector search
- LLM prompt template in spec: "Are these the same entity? Entity A: ... Entity B: ..."
- Estimated effort: 1 week
