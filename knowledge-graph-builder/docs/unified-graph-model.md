# Unified Source → Structure → Entity Graph Model

**Status:** v0.2 — TASK-221 (STORY-034). Revised after the SA review (2026-05-19).
**Implements:** ADR-022 D7. Resolves `wiki/5_research/data-graph-ingestion-survey.md` §3.3.

---

## 1. Purpose

Every data source — a database, a spreadsheet, a folder of documents, a code
repo — must turn into **one consistent kind of graph**. Today it does not:
files become `:Document` nodes, databases become `:Connector` nodes whose tables
are never modeled, and four ingestion paths each stamp their own provenance
fields. This document defines the single model that recipes (TASK-220) build.

**No backward compatibility.** The project is in development; the database is
reset to a clean state. This model is *the* model — there is no legacy model to
bridge or coexist with.

## 2. The core rule: one node per real thing

This is the decision the SA review asked for. Stated plainly:

> **A node is created for each real thing. Containers get their own node.
> The data *inside* a container does not get a second "structure" node — if
> it is a real thing, it becomes an entity node directly.**

Worked examples:

- **A database row** (an employee). The database is one `:Source` node. The
  `employees` table is one `:Table` node. Each employee **is one node** —
  `:Employee:__Entity__`. There is **no** separate "row node": the row *is* the
  employee. The employee node points back to its `:Table` so you can trace
  where it came from.
- **A code function.** The repo is one `:Source`. Each code file is one
  `:File`. Each function **is one node** — `:Function:__Entity__` — not a
  "symbol node" plus an "entity node".
- **A text document** — the one case where two nodes are right. The document is
  one `:Source`. It is split into `:Chunk` nodes (a chunk is a paragraph-ish
  unit — it is *not* itself a thing). The people, places, and systems *mentioned
  in* a chunk become their own entity nodes, linked back to the chunk.

So: **two nodes only when the container and its contents are genuinely
different things** (a chunk vs. the entities named in it). For a database or
code — where the row/function already *is* the thing — it is **one node**. This
avoids doubling the node count for the commonest data shapes.

## 3. Node families

| Family | Labels | What it is |
|---|---|---|
| **Source** | `:Source:__KGBuilder__` | One per ingested data source. |
| **Container** | `:Table`, `:Sheet`, `:File`, `:Chunk` — each `:__KGBuilder__` | A grouping *inside* a source. A fixed, platform-owned set — recipes never invent these. |
| **Entity** | `:__Entity__` + one recipe-supplied domain label (`:Employee`, `:Operator`, …) | A real thing. The domain type is a real Neo4j label, never a property. |

Domain labels are supplied by recipes and validated (safe-identifier +
ADR-015 reserved namespace — recipe-spec §5.5). The container label set is
fixed, so recipes cannot collide with it.

## 4. Identity — decoupled from storage primary keys

Per ADR-022 D7, an entity's identity does **not** come from its database primary
key — so the same real person appearing in two tables, or two sources, becomes
**one** node.

- **Source** — `source_id` = a deterministic hash of `(graph_id, source descriptor)`.
- **Container** — deterministic from its path inside the source.
- **Entity** — `id` = a deterministic hash of `(graph_id, domain label,
  normalized identity key)`. The identity key comes from the recipe's `identity`
  rule (recipe-spec §6).

**Identity-rule floor (SA finding 4).** A recipe's `identity` rule may only
normalize using the fixed set already enforced by the recipe JSON Schema —
`casefold`, `trim`, `collapse_whitespace`. A recipe **cannot** define an
arbitrary normalization, so it cannot over-aggressively collapse two distinct
real entities into one node. Identity is bounded by construction.

## 5. Provenance — one uniform set

Every node and edge the pipeline writes carries:

| Property | On | Meaning |
|---|---|---|
| `graph_id` | everything | tenant scope — always |
| `ingestion_source` | everything | the `source_id` it came from |
| `provenance` | everything | `EXTRACTED` (explicit in the source) or `INFERRED` |
| `confidence` | **`INFERRED` only** | a float `< 1.0`. Omitted on `EXTRACTED` — it would always be `1.0`, so storing it is waste (SA finding 7). |
| `recipe_id`, `recipe_version` | everything | the recipe that produced it |
| `ingestion_time` | everything | when written |

Provenance properties are **write-once**: the execution engine sets them when
the node/edge is first created; recipe re-runs update data via `MERGE` but do
not rewrite provenance. Full tamper-evidence against a malicious in-tenant actor
is **not** in this model — it is the same gap as the open
`AGENT-SCOPE-CONTAINMENT` concern and is out of scope here (SA finding 5).

## 6. Temporal model

Entity **edges** carry bitemporal `event_time` (when the fact was true) and
`transaction_time` (when it was recorded) where temporal correctness matters —
the bitemporal Commitment the platform already applies in `pipeline_service`.
`:Source` and container nodes carry only `ingestion_time`; they are not
bitemporal facts.

## 7. Relationships

- `(:Container)-[:PART_OF]->(:Source)` and `(:Container)-[:PART_OF]->(:Container)`
  — a table belongs to a source; a chunk belongs to a source.
- `(:__Entity__)-[:DERIVED_FROM]->(:Container | :Source)` — every entity links
  back to where it came from (a row's employee → its `:Table`; a chunk's
  extracted person → its `:Chunk`).
- `(:__Entity__)-[<domain type> { §5 provenance, §6 temporal }]->(:__Entity__)`
  — the semantic edges a recipe creates.

## 8. The migration

`app/cypher/migrations/2026-05-19_unified_graph_model.cypher` — idempotent
(`CREATE CONSTRAINT IF NOT EXISTS`). It establishes:

- a uniqueness constraint per node family (`:Source` on `(graph_id, source_id)`;
  `:__Entity__` and each container label on `(graph_id, id)`);
- a `graph_id` index per label, for tenant-scoped scans;
- an index on `:__Entity__(provenance)`.

It establishes the model on a clean database. Per the by-dependency testing
policy, it is applied and verified against a fresh Neo4j in the live Docker
stack.

## 9. Decisions taken (SA review, 2026-05-19)

| SA finding | Resolution |
|---|---|
| 1 — structure labels collide with existing `:Document`/`:Chunk`/`:File` | Moot: no backward compatibility, clean database. The container set in §3 is defined fresh. |
| 2 — separate Structure/Entity layers double the node count | Resolved by §2: one node per real thing. Fine-grain data (rows, functions) becomes an entity node directly — no paired structure node. |
| 3 — "single model" claim vs. legacy coexistence | Moot: no legacy. This *is* the single model (§1). |
| 4 — recipe identity rule could over-collapse entities | Resolved by §4: normalization is restricted to the recipe schema's fixed enum. |
| 5 — provenance not tamper-evident | §5: provenance is write-once; hardening against a malicious in-tenant actor is the open `AGENT-SCOPE-CONTAINMENT` concern, out of scope here. |
| 6 — cited the draft ADR-005 | §6 now cites the bitemporal Commitment / the existing `pipeline_service` behaviour, not ADR-005's status. |
| 7 — `confidence` constant on `EXTRACTED` | §5: `confidence` is stored only on `INFERRED` elements. |
