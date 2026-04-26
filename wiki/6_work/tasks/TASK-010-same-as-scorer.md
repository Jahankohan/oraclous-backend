---
id: TASK-010
title: "Implement multi-signal SAME_AS scorer and EntityResolver component"
story: STORY-005
assignee: backend-developer
reporter: reza
priority: high
status: in-review
created: 2026-04-26
updated: 2026-04-26
blocked_by: [TASK-009]
blocks: [TASK-011]
wiki_refs: []
estimated: "2d"
branch: "agent/STORY-005/TASK-010-same-as-scorer"
---

# TASK-010: Multi-Signal SAME_AS Scorer

## What

Create `app/components/entity_resolver.py` with a `EntityResolver` class that combines
four signals to produce a final confidence score for each SAME_AS candidate, then
creates `SAME_AS` links for candidates above the store threshold (0.85).

**Scoring formula:**
```
final_score = (
    embedding_similarity * 0.4 +
    name_similarity      * 0.3 +
    type_compatibility   * 0.2 +
    context_overlap      * 0.1
)
```

**Signal implementations:**

1. **Embedding similarity** (0.4): Cosine similarity between entity embeddings — already
   available from TASK-009's candidate retrieval (it's the raw vector score).

2. **Name similarity** (0.3): Jaro-Winkler similarity on normalized names.
   Normalization: lowercase, strip punctuation, strip legal suffixes (Inc., Ltd., Corp., LLC).
   Use `jellyfish.jaro_winkler_similarity(a, b)` — add `jellyfish` to requirements.txt.

3. **Type compatibility** (0.2):
   - Same type → 1.0
   - Compatible types (e.g., `Organization` ↔ `Company`) → 0.5 (define compatible pairs)
   - Different types → 0.0

4. **Context overlap** (0.1): Jaccard similarity of 1-hop neighbor entity names.
   ```python
   neighbors_a = set(get_neighbor_names(entity_a, graph_id_a))
   neighbors_b = set(get_neighbor_names(entity_b, graph_id_b))
   jaccard = len(neighbors_a & neighbors_b) / len(neighbors_a | neighbors_b)
   ```
   If either entity has no neighbors → score 0.0 (do not divide by zero).

**Thresholds:**
- `final_score >= 0.85` → create `SAME_AS` link (store)
- `0.60 <= final_score < 0.85` → candidate for LLM disambiguation (TASK-011)
- `final_score < 0.60` → discard

**Writing SAME_AS links:**
```cypher
MERGE (a)-[:SAME_AS {confidence: $score, method: 'multi-signal', created_at: datetime()}]->(b)
MERGE (b)-[:SAME_AS {confidence: $score, method: 'multi-signal', created_at: datetime()}]->(a)
```
(Bidirectional, idempotent via MERGE — same as existing exact-match code.)

**Files:**
- New: `app/components/entity_resolver.py` — `EntityResolver` class
- Modify: `app/services/federation_service.py` — call `EntityResolver` after TASK-009's candidate retrieval
- `requirements.txt` — add `jellyfish`

## Why

Four signals together handle the cases exact matching misses: embedding similarity catches
aliases, Jaro-Winkler handles partial names, type compatibility prevents incorrect merges,
and context overlap handles ambiguous names in different domains. Centralizing logic in
`EntityResolver` means it can also be used for intra-graph deduplication in future work.

## Scope

**In scope:**
- `EntityResolver` class with four-signal scoring
- Writing `SAME_AS` links for scores ≥ 0.85
- Passing candidates in [0.60, 0.85) to TASK-011
- `jellyfish` library for Jaro-Winkler

**Out of scope (explicit):**
- LLM disambiguation (TASK-011)
- Retroactive re-scoring of existing SAME_AS links
- Compatible type pairs beyond the obvious ones (Organization/Company, Person/Individual)

## Definition of Done

- [ ] `EntityResolver.score(entity_a, entity_b)` returns correct weighted score for a test pair
- [ ] Weights sum to 1.0 (unit test assertion)
- [ ] "IBM" ↔ "International Business Machines" produces `final_score >= 0.85` → SAME_AS link created
- [ ] "Apple Inc." (Organization) ↔ "Apple" (Fruit) produces `final_score < 0.60` → no SAME_AS link
- [ ] Zero-neighbor entity: context overlap = 0.0, no ZeroDivisionError
- [ ] SAME_AS links are bidirectional and idempotent (MERGE)
- [ ] Reviewed by: qa-engineer (TASK-012)

## Output

- `code`: `app/components/entity_resolver.py` (new), `app/services/federation_service.py`

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

### 2026-04-26 — backend-developer — open → in-progress

Starting implementation. Branch: `agent/STORY-005/TASK-010-same-as-scorer`.

### 2026-04-26 — backend-developer — in-progress → in-review

Implementation complete. Three commits:
- `requirements.txt`: added jellyfish>=1.0.1
- `app/components/entity_resolver.py` (new file): EntityResolver with four-signal scoring (embedding 0.4, Jaro-Winkler name 0.3, type compatibility 0.2, neighbor Jaccard 0.1); resolve_and_link() creates bidirectional SAME_AS for scores ≥ 0.85; logs ambiguous [0.60, 0.85) for TASK-011
- `app/services/federation_service.py`: added EntityResolver import and resolve_entity() method

### 2026-04-26 — security-architect — security review: BLOCKED

See `wiki/4_agents/security-findings.md` — TASK-010 section.

Blocking (must fix before QA):
- Finding 4 (critical): apoc.cypher.doIt() + rel_type concatenation in MultiTenantEntityDeduplicator._create_relationship_fallback() — Cypher injection. Delete apoc.cypher.doIt() block; replace with allowlist dispatch.
- Finding 5 (high): source_graph_id fallback to target_graph_ids[0] in resolve_and_link() — cross-tenant neighbor read. Remove fallback; skip candidates with missing source_graph_id.
- Finding 6 (high): _create_same_as_link() MATCH clauses have no graph_id constraint; resolve_entity() has no _validate_and_filter() call. Add {graph_id} to both MATCH clauses; add _validate_and_filter() at top of resolve_entity().

Low (fix required per Reza decision 2026-04-26):
- Finding 7 (low): no name length cap before Jaro-Winkler — CPU DoS via long names. Add name = name[:1000] in _normalize_name().

### 2026-04-26 — backend-developer — fix applied

Applied all security fixes from TASK-010 review:
- entity_resolver.py: deleted apoc.cypher.doIt() block in _create_relationship_fallback(); replaced with _ALLOWED_REL_TYPES frozenset allowlist dispatch — zero executable Cypher from runtime values
- entity_resolver.py: removed source_graph_id fallback to target_graph_ids[0] in resolve_and_link(); malformed candidates now skipped with warning
- entity_resolver.py + federation_service.py: added graph_id_a/graph_id_b params to _create_same_as_link() MATCH clauses; added user_id + principal + _validate_and_filter() to resolve_entity()
- entity_resolver.py: added _MAX_NAME_LEN = 1000 cap in _normalize_name() before regex processing

### 2026-04-26 — security-architect — re-review: PASS

All four blocking findings resolved; two new non-blocking observations noted (allowlist coverage gap vs. extraction prompt types — QA defect; _store_same_as_links pre-existing no-graph_id pattern — pre-existing gap). Task cleared for QA sign-off.

### 2026-04-26 — backend-developer — functional fix: expand _ALLOWED_REL_TYPES

Updated _ALLOWED_REL_TYPES frozenset in entity_resolver.py to cover all relationship types
produced by the LLM extractor in pipeline_service.py. Previous list only covered WORKS_FOR;
the other 10 extractor types would have been silently dropped in the _create_relationship_fallback()
APOC fallback path. Defect identified during security re-review (Finding N1).
