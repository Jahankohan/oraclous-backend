---
id: STORY-004
title: "Add data flow analysis to Code KG: FLOWS_TO edges and taint tracking"
type: feature
layer: knowledge-graph
reporter: reza
status: ready
priority: medium
created: 2026-04-26
updated: 2026-04-26
wiki_refs: ["layer-1-knowledge-graph"]
tasks: []
decisions: []
---

# STORY-004: Code KG Data Flow Analysis

## Summary

The Code KG extracts a static call graph (CALLS, IMPORTS, INHERITS_FROM edges) via
Tree-sitter AST parsing for 5 languages. This answers structural questions ("what calls
this function?") but not semantic questions ("where does user input end up?"). Data flow
analysis adds `FLOWS_TO` edges tracking how parameters flow through assignments, returns,
and cross-function calls — enabling security analysis (taint tracking) and understanding
side effects. Phase 1 covers intra-procedural Python data flow.

## Problem Statement

- Code KG can answer "what does function X call?" but not "what happens to user input in X?"
- No `FLOWS_TO` edges exist; taint sources are unmarked
- Security queries (data reaches DB, user input reaches external API) are impossible
- Research (CGM, CODEXGRAPH) shows data flow is where code KG's real value lies

## Goals

- [ ] Implement `DataFlowAnalyzer` for Python functions: track parameter → local variable → return value → argument flows
- [ ] Produce `FLOWS_TO` edges in Neo4j with `via` property (assignment, argument, return)
- [ ] Mark taint sources (function parameters receiving HTTP request data, external API responses) with `taint: "user_input"` property
- [ ] Add data flow query endpoint to `app/api/v1/endpoints/code.py`
- [ ] Phase 1: Python only; TypeScript/Go/Java in follow-up stories

## Non-Goals

- Inter-procedural data flow (cross-function taint propagation) — Phase 2 follow-up
- Cross-repo linking — Phase 3 follow-up (requires federation)
- Type flow analysis

## Acceptance Criteria

- [ ] `analyze_function` produces `FLOWS_TO` edges for a Python function with ≥2 parameter assignments
- [ ] `FLOWS_TO {via: "argument"}` edge created when a tracked variable is passed to an external call
- [ ] Cypher query `MATCH (source {taint: "user_input"})-[:FLOWS_TO*1..10]->(sink)` returns correct path for a test function
- [ ] Data flow analysis completes for a 500-line Python file in <10s
- [ ] Unit tests: parameter → local var, local var → return, local var → argument

## Open Questions

| # | Question | Owner | Status |
|---|----------|-------|--------|
| 1 | Which Tree-sitter query patterns cover Python assignment, call args, return? | engineering | open |

## Context & Background

- Full technical spec: `ORACLOUS_DEEPENING_ROADMAP.md` § 8 (Code KG Data Flow, pp. 522-605)
- Current impl: `app/services/code_parser_service.py`; `app/api/v1/endpoints/code.py`
- Tree-sitter AST already extracts `raw_calls`, `raw_imports` — build on this
- Estimated effort: 2-3 weeks (Phase 1 Python only)
