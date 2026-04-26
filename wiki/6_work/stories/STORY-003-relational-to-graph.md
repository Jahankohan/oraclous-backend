---
id: STORY-003
title: "Complete relational-to-graph transformation: schema mapper and row transformer"
type: feature
layer: knowledge-graph
reporter: reza
status: ready
priority: high
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-1-knowledge-graph"]
tasks: []
decisions: []
---

# STORY-003: Relational-to-Graph Inference

## Summary

The database connector captures relational schema metadata (`ColumnMeta` with FK references,
`SchemaSnapshot`) but does not transform rows into knowledge graph entities and
relationships. The connector's core value proposition — "connect your database and get a
knowledge graph" — is undelivered. This story implements `SchemaMapper` (table analysis →
graph mapping rules) and `RowTransformer` (rows → entity/relationship writes).

## Problem Statement

- `database_connector_service.py` introspects schema but produces no graph nodes
- Junction tables, self-referential FKs, and composite keys are not handled
- Sync modes exist (`full_snapshot`, `schema_only`) but `full_snapshot` writes nothing useful
- The connector is a metadata reader, not a graph builder

## Goals

- [ ] Implement `SchemaMapper` that classifies tables as entity tables vs junction tables and maps FK columns to relationship types
- [ ] Implement `RowTransformer` that converts entity table rows to `__Entity__` nodes and junction table rows to relationships
- [ ] Handle self-referential FKs (e.g., `employee.manager_id → employee.id`)
- [ ] Handle composite FKs and multi-column relationships
- [ ] Add optional LLM-assisted semantic naming for ambiguous schemas (junction table `employee_project` → `WORKS_ON`)
- [ ] Wire `RowTransformer` into the `full_snapshot` Celery sync task

## Non-Goals

- Cross-database join inference (relationships across different DB instances)
- Real-time CDC (change data capture); sync-based polling is sufficient for now
- Automatic schema evolution detection

## Acceptance Criteria

- [ ] A PostgreSQL table with PK + non-FK columns produces `__Entity__` nodes with correct properties
- [ ] A junction table (2 FK columns) produces relationship edges, not entity nodes
- [ ] A self-referential FK (`employee.manager_id`) produces a `MANAGES` or `REPORTS_TO` relationship
- [ ] Full sync of a 1000-row table completes without timeout under default Celery task settings
- [ ] LLM naming prompt produces human-readable relationship types for 5/5 test junction tables
- [ ] Unit tests: junction table detection, self-referential FK, row → entity mapping

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | LLM naming: synchronous during sync task or async best-effort? | engineering | open |

## Context & Background

- Full technical spec: `ORACLOUS_DEEPENING_ROADMAP.md` § 7 (Relational-to-Graph, pp. 442-518)
- Current impl: `app/services/database_connector_service.py`; `app/services/connector_jobs.py`
- Pipeline integration: `app/services/pipeline_service.py` (pre-structured entity path)
- Estimated effort: 1-2 weeks
