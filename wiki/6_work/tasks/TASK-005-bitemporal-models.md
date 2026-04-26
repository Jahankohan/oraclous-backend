---
id: TASK-005
title: "Add bitemporal properties to entity/relationship models and migrate existing data"
story: STORY-002
assignee: backend-developer
reporter: reza
priority: critical
status: in-review
created: 2026-04-26
updated: 2026-04-26
blocked_by: []
blocks: [TASK-006, TASK-007]
wiki_refs: ["layer-1-knowledge-graph"]
estimated: "2d"
branch: "agent/STORY-002/TASK-005-bitemporal-models"
---

# TASK-005: Bitemporal Property Schema and Migration

## What

Add four new properties to every entity node and relationship in Neo4j, and migrate
existing data so nothing is lost.

**Properties to add:**
```
event_time: datetime       -- when the fact was true in the real world (nullable)
event_time_end: datetime   -- when the fact stopped being true (null = still true)
ingestion_time: datetime   -- when the system first learned this fact
ingestion_source: string   -- which document/connector/API provided it
```

**Migration strategy for existing nodes:**
```cypher
// Run once per graph (or across all graphs as admin)
MATCH (e:__Entity__)
WHERE e.event_time IS NULL
SET e.event_time = e.ingested_at,
    e.ingestion_time = e.ingested_at,
    e.ingestion_source = 'pre-migration'

MATCH ()-[r]->()
WHERE r.event_time IS NULL AND r.ingested_at IS NOT NULL
SET r.event_time = r.ingested_at,
    r.ingestion_time = r.ingested_at,
    r.ingestion_source = 'pre-migration'
```

**Where to run migration:**
- As a Celery beat one-off task (add to `background_jobs.py` with a guard: skip if
  already run, checked via a flag node `(:Migration {id: 'bitemporal-v1', done: true})`)
- Alternatively as a startup migration in `app/core/startup.py` if that pattern exists

**Model updates:**
- Update `app/models/` entity and relationship Pydantic models to include the four new
  fields (all optional/nullable for backward compatibility)
- Update any Cypher `CREATE` and `MERGE` statements that write entities to include
  `ingestion_time: datetime()` and `ingestion_source: $source` parameters

**The `valid_from` / `valid_to` fields stay unchanged** — they are used by Ebbinghaus
decay and are separate from temporal validity. Do not remove them.

## Why

This is the foundation for TASK-006 (LLM extraction) and TASK-007 (temporal query modes).
Without `event_time` and `ingestion_time` being distinct properties in the schema, the
temporal query filters in TASK-007 have nothing to operate on.

## Scope

**In scope:**
- Adding four properties to entity and relationship nodes
- Writing and running the migration (backward-compatible; existing nodes get event_time = ingested_at)
- Updating Pydantic models to include new fields (nullable)
- Updating entity CREATE/MERGE Cypher to write `ingestion_time` and `ingestion_source`

**Out of scope (explicit):**
- LLM extraction changes (TASK-006)
- Chat service query modes (TASK-007)
- Changing or removing `valid_from` / `valid_to` (Ebbinghaus fields — do not touch)
- Retroactively improving event_time accuracy for historical data

## Definition of Done

- [ ] All `__Entity__` nodes and relationship records have `event_time`, `ingestion_time`, `ingestion_source` properties after migration
- [ ] Migration is idempotent: running it twice produces no errors and no duplicate data
- [ ] Migration guard prevents re-run on subsequent restarts
- [ ] New entity writes include `ingestion_time = datetime()` and `ingestion_source` parameter
- [ ] Pydantic models accept `event_time`, `event_time_end`, `ingestion_time`, `ingestion_source` as optional fields
- [ ] Existing entity read/write tests still pass (no regressions)
- [ ] Reviewed by: qa-engineer (TASK-008)

## Output

- `code`: `app/models/`, `app/services/pipeline_service.py` (Cypher write statements), `app/services/background_jobs.py` (migration task)

## Notes / Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

## Agent Log

Append-only. See [AGENT_PROTOCOL.md](../AGENT_PROTOCOL.md) for format rules.

### 2026-04-26 — backend-developer — open → in-progress

Starting implementation. Branch: `agent/STORY-002/TASK-005-bitemporal-models`.

### 2026-04-26 — backend-developer — in-progress → in-review

Implementation complete. Three commits:
- `graph_schemas.py`: added four bitemporal fields (event_time, event_time_end, ingestion_time, ingestion_source) to EntityNodeProperties and RelationshipProperties as Optional fields
- `multi_tenant_components.py`: MultiTenantKGWriter accepts ingestion_source; sets ingestion_time via datetime.now(UTC) and ingestion_source via setdefault in run()
- `background_jobs.py`: added run_bitemporal_migration_v1 Celery task with idempotency guard (:Migration {id: 'bitemporal-v1', done: true})

Decision: used Celery beat one-off pattern rather than startup migration to avoid blocking startup on large graphs.

### 2026-04-26 — security-architect — security review: BLOCKED

See `wiki/4_agents/security-findings.md` — TASK-005 section.

Blocking (must fix before QA):
- Finding 1 (high): migration queries have no graph_id filter — cross-tenant bulk write. Fan-out per graph_id required.

Medium (fix required per Reza decision 2026-04-26):
- Finding 2 (medium): ingestion_source setdefault preserves LLM-extracted value — prompt injection vector. Pop/sanitize with 512-char limit.
- Finding 3 (medium): ingestion_source has no max_length; ingestion_time has no range bound. Add Field(max_length=512) + field_validator.

Non-blocking noted:
- Finding 4 (informational): add comment explaining graph_id .update() asymmetry.
- Finding 7 (low): TOCTOU race in migration guard; use atomic MERGE ON CREATE.

### 2026-04-26 — backend-developer — fix applied

Applied all security fixes from TASK-005 review:
- background_jobs.py: fan-out migration per graph_id; atomic MERGE guard with graph-scoped id; _run_bitemporal_migration_for_graph() handles single graph
- multi_tenant_components.py: pop/sanitize ingestion_source; _sanitize_source() helper; added comment on graph_id .update() security rationale
- graph_schemas.py: ingestion_source max_length=512 via Field(); ingestion_time field_validator rejects values outside [2020-01-01, now+1h]

### 2026-04-26 — security-architect — re-review: STILL BLOCKED

Fixes claimed in agent log are not present in code. All three findings remain unresolved in the on-disk files. See `wiki/4_agents/security-findings.md` — TASK-005 re-review section.

### 2026-04-26 — security-architect — re-review (corrected): PASS

Previous re-review read the outer monorepo repo (no code tracked there) — not the inner oraclous-data-studio repo. Re-reviewed from correct repo: all three findings are resolved. See `wiki/4_agents/security-findings.md` — TASK-005 corrected re-review section.
