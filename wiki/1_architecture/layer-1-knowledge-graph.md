# Layer 1: Knowledge Graph

**Layer:** Knowledge Graph
**Status:** Phases 1-4 complete; Deepening phase (April–July 2026) in progress
**Authority:** ADR-001 through ADR-005 govern this layer

---

## What Layer 1 Is

The multi-tenant knowledge graph substrate. Every other layer sits on top of it.
L1 stores everything: entities, relationships, agent memories, checkpoints, audit events,
HITL review nodes, community structures, versioned snapshots, federation links.

L1 is the coordination medium, the shared memory, and the audit log. It is not a
cache or a secondary store — it is the authoritative source of truth (ADR-005).

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1: Knowledge Graph (Neo4j)                                │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Graph Store  │  │  ReBAC Auth  │  │  Zero-Copy Versions  │  │
│  │  Multi-tenant │  │  Phase B     │  │  Snapshot + Diff     │  │
│  │  graph_id on  │  │  13 perms    │  │  transaction_time    │  │
│  │  every query  │  │  5 roles     │  │  invalidated_at      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Federation  │  │  Agent Layer │  │  Community Detection  │  │
│  │  SAME_AS     │  │  Memory      │  │  Leiden hierarchical  │  │
│  │  UNION ALL   │  │  Checkpoints │  │  multi-level summaries│  │
│  │  fail-closed │  │  HITL gates  │  │  global search path  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Bitemporal  │  │  Code KG     │  │  Evaluation          │  │
│  │  event_time  │  │  AST + data  │  │  RAGAS endpoint      │  │
│  │  ingestion_t │  │  flow (P1)   │  │  test set storage    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## What Exists (Phases 1-4 Complete)

| Capability | Status | Key Files |
|---|---|---|
| Multi-tenant graph store | Done | Every Cypher query has `graph_id` filter |
| ReBAC Phase B | Done | 13 system permissions, 5 roles, role inheritance, Redis-cached (60s TTL) |
| Zero-copy versioning | Done | `transaction_time` / `invalidated_at` soft-delete; snapshot + diff + rollback |
| Cross-graph federation | Done | Fail-closed UNION ALL routing; SAME_AS exact-match (MVP) |
| Service accounts | Done | JWT, key rotation, cross-graph grants |
| Agent memory (Ebbinghaus) | Done | Decay: `I(t) = base_importance × e^(-λ×days) + access_boost` |
| Community detection (Louvain) | Done | Multi-level (1-5), LLM summaries, staleness tracking |
| Code KG (AST) | Done | Tree-sitter, 5 languages, CALLS/IMPORTS/INHERITS/MEMBER_OF/DEFINED_IN |
| Database connectors | Done | SQL, Sheets, API, webhook; schema introspection; SSRF guards |
| Evaluation (RAGAS) | Done | Faithfulness, relevance, precision, recall as REST endpoint |
| MCP server (L1 tools) | Done | 15 tools; some gaps per ADR-007 audit |
| OpenTelemetry | Done | OTEL instrumentation; connection pool gaps (see STORY-009) |
| Security hardening | Done | Parameterized Cypher everywhere; rate limiting; 403 info-leakage fixed; SSRF guards |

---

## What Is Being Deepened (April–July 2026)

These are correctness and quality upgrades to existing features — not new features.

| Work Stream | What's Incomplete | Story |
|---|---|---|
| **WS1: Leiden Communities** | `leidenalg` in requirements.txt but GDS Louvain used; hierarchy not wired into retrieval | [STORY-001](../6_work/stories/STORY-001-leiden-communities.md) |
| **WS2: Bitemporal Tracking** | `valid_from` conflated with ingestion time; no `event_time` / `event_time_end` separation | [STORY-002](../6_work/stories/STORY-002-bitemporal-tracking.md) |
| **WS3: Relational-to-Graph** | Schema captured, rows not transformed to entities/relationships | [STORY-003](../6_work/stories/STORY-003-relational-to-graph.md) |
| **WS4: Code KG Data Flow** | AST only; no FLOWS_TO edges, no taint tracking, no cross-repo linking | [STORY-004](../6_work/stories/STORY-004-code-kg-data-flow.md) |
| **WS5: Federation SAME_AS** | Exact name+type match only; thresholds defined but embedding similarity not used | [STORY-005](../6_work/stories/STORY-005-federation-same-as.md) |
| **WS6: Multimodal Depth** | No OCR fallback for scanned PDFs; no diagram understanding; no CSV/JSON/MD | [STORY-006](../6_work/stories/STORY-006-multimodal-depth.md) |
| **WS7: Frontend MVP** | React scaffold exists; no backend integration; platform is user-unreachable | [STORY-007](../6_work/stories/STORY-007-frontend-mvp.md) |
| **WS8: Benchmarks** | Zero published performance or quality numbers exist | [STORY-008](../6_work/stories/STORY-008-benchmarks.md) |
| **WS9: Production Hardening** | Query caching, connection pool metrics, graceful degradation, rate limit refinement | [STORY-009](../6_work/stories/STORY-009-production-hardening.md) |

---

## Key Invariants (Never Violated)

These properties hold across all L1 code and are enforced by the Scope Enforcer at L2:

1. **`graph_id` on every Cypher query** — no cross-tenant reads ever
2. **Parameterized Cypher everywhere** — no string interpolation of user input
3. **Fail-closed security** — deny by default (federation, ReBAC, service accounts)
4. **Dual driver pattern** — `neo4j_client.async_driver` for FastAPI; NullPool sync driver for Celery tasks
5. **One service per major functionality** — no `enhanced_*` duplicate files

---

## Node Labels Used by L2

These L1 node types are created and read by the L2 Harnessing Platform:

| Label | Purpose | Key Properties |
|---|---|---|
| `:AgentTask` | Agent execution unit; created by APOC trigger | `id`, `graph_id`, `agent_type`, `status`, `created_at` |
| `:AgentRun` | Execution record of an agent invocation | `run_id`, `graph_id`, `agent_type`, `state`, `started_at` |
| `:AgentMemory` | Bitemporal agent memory | `id`, `agent_id`, `graph_id`, `key`, `value`, `valid_time`, `transaction_time` |
| `:Checkpoint` | BSP super-step state | `run_id`, `graph_id`, `super_step`, `state_json`, `ttl` |
| `:HITLReview` | Human-in-the-loop gate | `id`, `run_id`, `graph_id`, `status`, `timeout_at`, `resolution` |
| `:AuditEvent` | Immutable audit record | `id`, `graph_id`, `event_type`, `actor`, `hash`, `prev_hash` |

---

## ADRs Governing This Layer

| ADR | Decision |
|---|---|
| [ADR-001](../3_decisions/ADR-001-rebac-new-graphs-only.md) | ReBAC applied to new graphs only |
| [ADR-002](../3_decisions/ADR-002-cross-graph-federation-opt-in.md) | Federation is opt-in |
| [ADR-003](../3_decisions/ADR-003-versioning-snapshot-approach.md) | Versioning via zero-copy snapshot |
| [ADR-004](../3_decisions/ADR-004-agent-subgraph-access-model.md) | Agent subgraph access model |
| [ADR-005](../3_decisions/ADR-005-l1-sole-persistence-layer.md) | L1 is sole persistence layer |

---

## Source of Full Technical Detail

The deepening work streams are fully spec'd in:
`ORACLOUS_DEEPENING_ROADMAP.md` — April 12, 2026 document with current state, required changes, file-level targets, and test requirements for each work stream.
