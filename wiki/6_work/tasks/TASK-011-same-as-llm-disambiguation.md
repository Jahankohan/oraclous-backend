---
id: TASK-011
title: "Add LLM disambiguation for ambiguous SAME_AS candidates (score 0.60-0.85)"
story: STORY-005
assignee: ai-agent-specialist
reporter: reza
priority: high
status: in-progress
created: 2026-04-26
updated: 2026-04-26
blocked_by: [TASK-010]
blocks: [TASK-012]
wiki_refs: []
estimated: "1-2d"
branch: "agent/STORY-005/TASK-011-same-as-llm-disambiguation"
---

# TASK-011: LLM SAME_AS Disambiguation

## What

For candidates where `0.60 <= multi_signal_score < 0.85` (ambiguous zone), call an LLM
to resolve whether the two entities are the same. Clear yes → create SAME_AS. Clear no → discard.

**Disambiguation prompt** (add to `app/services/llm_service.py`):
```
Are these two entities the same real-world entity? Answer YES or NO, and provide a brief reason.

Entity A: "{name_a}" (type: {type_a})
Context A: {context_a}    <- up to 3 neighboring entity names

Entity B: "{name_b}" (type: {type_b})
Context B: {context_b}

Answer format:
DECISION: YES | NO
CONFIDENCE: HIGH | MEDIUM | LOW
REASON: <one sentence>
```

**Handling the response:**
- `YES` + `HIGH` confidence → create SAME_AS with `method: "llm-disambiguated"`, `confidence: multi_signal_score`
- `YES` + `MEDIUM` confidence → create SAME_AS with `confidence: multi_signal_score * 0.9`
- `YES` + `LOW` or `NO` → discard candidate (no SAME_AS link)

**Where to call this:**
- In `EntityResolver` (from TASK-010), after scoring: pass ambiguous candidates to a new
  `_llm_disambiguate(entity_a, entity_b)` method
- Call `llm_service.py`'s disambiguation function (add a new function there — do not reuse chat functions)

**Rate limiting**: LLM disambiguation is called only for ambiguous candidates. Expect
few calls per federation run for typical datasets — no batch rate limit concern.

**Files to modify:**
- `app/components/entity_resolver.py` — add `_llm_disambiguate()` method (calls llm_service)
- `app/services/llm_service.py` — add `disambiguate_entities(entity_a, entity_b, context_a, context_b)` function

## Why

Multi-signal scoring cannot resolve all ambiguities — "J. Smith" vs "John Smith" may
score in the ambiguous range (good Jaro-Winkler, moderate context overlap). LLM
disambiguation handles the hard cases without creating false positives from over-eager
scoring.

## Scope

**In scope:**
- Disambiguation prompt and LLM call for candidates in [0.60, 0.85)
- Writing SAME_AS with `method: "llm-disambiguated"` for confirmed matches
- Discarding candidates on LLM NO or LOW confidence

**Out of scope (explicit):**
- Batch LLM calls (sequential is fine; disambiguation runs rarely)
- LLM provider selection (uses whatever llm_service.py uses — no gateway yet; this changes when STORY-011 is done)
- Caching LLM disambiguation results (follow-up optimization)

## Definition of Done

- [ ] "J. Smith" ↔ "John Smith" in same domain context: LLM returns YES → SAME_AS created
- [ ] "Apple Inc." (company context) ↔ "Apple" (fruit context): LLM returns NO → no SAME_AS
- [ ] Disambiguation prompt is logged at DEBUG level for observability
- [ ] LLM call only happens for scores in [0.60, 0.85) — not for scores ≥ 0.85 or < 0.60
- [ ] Unit test: mock LLM returning YES/HIGH → SAME_AS created; mock returning NO → not created
- [ ] Reviewed by: qa-engineer (TASK-012)

## Output

- `code`: `app/components/entity_resolver.py`, `app/services/llm_service.py`

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

### 2026-04-26 — ai-agent-specialist — open → in-progress

Starting implementation. Branch: `agent/STORY-005/TASK-011-same-as-llm-disambiguation`.
Adding disambiguate_entities() to llm_service.py and _llm_disambiguate() to EntityResolver in entity_resolver.py.
