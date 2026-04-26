---
id: TASK-009
title: "Replace exact SAME_AS matching with embedding-based candidate retrieval"
story: STORY-005
assignee: backend-developer
reporter: reza
priority: high
status: in-review
created: 2026-04-26
updated: 2026-04-26
blocked_by: []
blocks: [TASK-010]
wiki_refs: ["layer-1-knowledge-graph"]
estimated: "2d"
branch: "agent/STORY-005/TASK-009-same-as-embeddings"
---

# TASK-009: Embedding-Based SAME_AS Candidate Retrieval

## What

Replace the current exact name+type match in `app/services/federation_service.py`
(lines 353-376) with a vector-search-based candidate retrieval step.

**Current code** (exact match, produces confidence 0.99 for any exact hit):
```python
# Locate: exact name + type match in federation_service.py
# Replace with the approach below
```

**New approach:**

1. **Get the entity's embedding**: `__Entity__` nodes already have embeddings stored.
   Retrieve the embedding for the entity being matched.

2. **Vector search over target graphs**: Run a vector similarity search over `__Entity__`
   nodes in the target graph_ids. Use the existing Neo4j vector index (it already exists
   for chat retrieval — reuse it). Filter by `candidate_threshold = 0.60`.

3. **Return candidates**: Return all entities above the candidate threshold as
   `SameAsCandidate` objects with their raw cosine similarity score. Do NOT create
   `SAME_AS` links in this step — that is TASK-010's job.

4. **Preserve exact match as a fast path**: If an exact name+type match exists (same
   name, same type, confidence 0.99), return it directly without vector search. The
   vector search is the fallback for non-exact cases.

```python
async def find_same_as_candidates(
    self, entity: dict, target_graph_ids: list[str]
) -> list[SameAsCandidate]:
    # 1. Fast path: exact name+type
    exact = self._find_exact_match(entity, target_graph_ids)
    if exact:
        return [SameAsCandidate(entity=exact, score=0.99, method="exact")]

    # 2. Vector search
    embedding = entity.get("embedding")
    if not embedding:
        return []
    candidates = await self._vector_search(embedding, target_graph_ids, threshold=0.60)
    return [SameAsCandidate(entity=c, score=c["similarity"], method="vector") for c in candidates]
```

**Files to modify:**
- `app/services/federation_service.py` — replace `find_same_as_candidates` implementation

## Why

Exact matching misses aliases, partial names, and contextual entities — the three main
federation failure modes. Embedding-based retrieval is the first step; TASK-010 adds
multi-signal scoring on top of these candidates to decide which ones become SAME_AS links.

## Scope

**In scope:**
- Replacing exact-only match with vector search + exact fast path
- `SameAsCandidate` dataclass if not already defined
- Using the existing vector index on `__Entity__` nodes

**Out of scope (explicit):**
- Multi-signal scoring (TASK-010)
- LLM disambiguation (TASK-011)
- Creating `SAME_AS` links (TASK-010)
- Adding embeddings to entities that don't have them (pre-existing gap; skip entities with no embedding)

## Definition of Done

- [ ] `find_same_as_candidates` returns results for "International Business Machines" when given an "IBM" entity (same graph, separate entities for testing purposes)
- [ ] Returns empty list for entities with no embedding (no crash)
- [ ] Exact match fast path still returns confidence 0.99 for exact hits
- [ ] Candidate threshold 0.60 is applied — no candidates below this score returned
- [ ] Unit test: vector search returns candidates sorted by similarity descending
- [ ] Reviewed by: qa-engineer (TASK-012)

## Output

- `code`: `app/services/federation_service.py`

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

### 2026-04-26 — backend-developer — open → in-progress

Starting implementation. Branch: `agent/STORY-005/TASK-009-same-as-embeddings`.

### 2026-04-26 — backend-developer — in-progress → in-review

Implementation complete. Two commits:
- `federation_schemas.py`: added SameAsCandidate TypedDict
- `federation_service.py`: added find_same_as_candidates(), _find_exact_match(), _vector_search_candidates(); uses entity_embeddings vector index; _SAME_AS_CANDIDATE_THRESHOLD = 0.60

### 2026-04-26 — security-architect — security review: BLOCKED

See `wiki/4_agents/security-findings.md` — TASK-009 section.

Blocking (must fix before QA):
- Finding 1 (high): find_same_as_candidates() has no auth gate — no _validate_and_filter() call. Add user_id + principal params, call _validate_and_filter() first.
- Finding 2 (high): candidate_count unbounded — DoS via large target_graph_ids. Add MAX_TARGET_GRAPHS guard.

Medium (fix required per Reza decision 2026-04-26):
- Finding 3 (medium): no embedding dimensionality check or float coercion. Validate len == 3072, coerce to float.

### 2026-04-26 — backend-developer — fix applied

Applied all security fixes from TASK-009 review:
- federation_service.py: added user_id + principal params to find_same_as_candidates(); calls _validate_and_filter() before any Neo4j access, matching the pattern of all other public methods
- federation_service.py: added _MAX_TARGET_GRAPHS = 50 module constant; guard at top of _vector_search_candidates() raises ValueError if exceeded
- federation_service.py: added _EMBEDDING_DIM = 3072 module constant; dimensionality check and float coercion before vector index call
