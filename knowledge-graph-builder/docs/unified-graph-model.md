# Unified Source → Structure → Entity Graph Model

**Status:** v0.1 draft — TASK-221 (STORY-034). Pending Solution Architect review.
**Implements:** ADR-022 D7. Resolves the inconsistencies in
`wiki/5_research/data-graph-ingestion-survey.md` §3.3–§3.4.

---

## 1. Purpose

Every data source — relational, structured, text, code — must project into one
**consistent** graph model. Today it does not: files become `:Document` nodes,
databases become `:Connector` nodes whose tables/keys are never modeled, and
four ingestion paths stamp four different provenance vocabularies (survey §3.3).
This document defines the single model recipes (TASK-220) project into.

## 2. Three layers

```
Source        — one node per ingested data source
   ↑ PART_OF
Structure     — the source's structural units (tables, columns, records, …)
   ↑ DERIVED_FROM
Entity        — the semantic entities a recipe projects
```

- **Source** — `:Source:__KGBuilder__`. One per ingested source. Properties:
  `source_id`, `source_type`, `name`, `shape_signature`, `recipe_id`,
  `recipe_version`, `graph_id`, `ingestion_time`.
- **Structure** — per-kind labels from a **governed, fixed set**, each also
  carrying `:__KGBuilder__`: `:Table`, `:Column`, `:Sheet`, `:Record`,
  `:Chunk`, `:Symbol`. Recipes never invent structure labels — the set is
  platform-fixed, which is what prevents a recurrence of the `:Module` →
  `:CodeModule` collision (ADR-015).
- **Entity** — `:__Entity__` plus exactly **one recipe-supplied domain label**
  (`:Employee`, `:Operator`, …), validated per recipe-spec §5.5. The domain
  type is a real Neo4j label, never a property.

## 3. The §3.3 inconsistencies — resolved

| # | Inconsistency today | Resolution |
|---|---|---|
| 1 | `__Entity__` used three ways; relational rows hide the type in a property | Every entity = `:__Entity__` + one real domain label. No label-as-property. |
| 2 | Four incompatible ID schemes (`entity_id`, `qualified_name`, `{connector}:{table}:{pk}`, record-`id`) | One scheme — see §4. Identity is decoupled from storage primary keys. |
| 3 | Four divergent provenance vocabularies | One uniform provenance property set — see §5. |
| 4 | Temporal model is doc-only | Bitemporal properties on entity edges — see §6. |
| 5 | No governed label namespace (the `:Module` collision) | Structure labels are a fixed reserved set; domain labels are recipe-supplied + validated. |
| 6 | Confidence / audit exists only in `graphify`, not the backend | `provenance` + `confidence` on every node and edge — §5. |

## 4. Identity

Decoupled from storage primary keys (ADR-022 D7), so the same real-world entity
converges across rows, columns, and sources.

- **Source** — `source_id` = deterministic hash of `(graph_id, source descriptor)`.
- **Structure** — deterministic from the unit's path within its source:
  `(graph_id, source_id, structural path)` (e.g. `source · table · column`).
- **Entity** — `id` = deterministic hash of `(graph_id, label, normalized
  identity key)`, where the normalized key comes from the recipe's `identity`
  rule (recipe-spec §6). **Never the storage PK.**

Uniqueness constraints: `:Source` on `(graph_id, source_id)`; each structure
label on `(graph_id, id)`; `:__Entity__` on `(graph_id, id)`.

## 5. Provenance — one uniform set

Every node and every edge the ingestion pipeline writes carries:

| Property | Meaning |
|---|---|
| `graph_id` | tenant scope — on everything, always |
| `ingestion_source` | the `source_id` it came from |
| `provenance` | `EXTRACTED` (explicit in the source) or `INFERRED` (recipe-spec §8) |
| `confidence` | float; `1.0` for `EXTRACTED`, `<1.0` for `INFERRED` |
| `recipe_id`, `recipe_version` | the recipe that produced it |
| `ingestion_time` | when written |

This replaces the four divergent vocabularies and gives the backend the
`graphify` provenance-of-belief model it currently lacks.

## 6. Temporal model

Entity **edges** carry bitemporal `event_time` and `transaction_time` where
temporal correctness matters (ADR-005 / Commitment 6). `:Source` and structure
nodes carry `ingestion_time` only — they are not bitemporal facts.

## 7. Relationships

- `(:Structure)-[:PART_OF]->(:Source)` and `(:Structure)-[:PART_OF]->(:Structure)`
  — containment (a column part-of a table, a chunk part-of a document).
- `(:__Entity__)-[:DERIVED_FROM]->(:Structure)` — every entity links back to the
  structural unit it was projected from (provenance-of-belief, ADR-022 D6/D8).
- `(:__Entity__)-[<domain type> { …§5 provenance, §6 temporal }]->(:__Entity__)`
  — the semantic edges a recipe creates.

## 8. The migration (to be written + tested next)

A `.cypher` migration under `app/cypher/migrations/`, idempotent
(`CREATE CONSTRAINT IF NOT EXISTS`), establishing: the uniqueness constraints of
§4, a `graph_id` index per label, and an index on `provenance`. It establishes
the *target* model; it does not migrate existing data. Per the by-dependency
testing policy, it is tested against a fresh Neo4j in the **live Docker stack**.

## 9. Open questions — for the `/sa` review

1. **Legacy coexistence.** Existing `:Document` / `:Connector` / `:CodeModule`
   nodes are *not* migrated by this task (out of scope). So the unified model
   coexists with the legacy model until a separate reconciliation. Is that
   acceptable, or must TASK-221 also alias/bridge the legacy labels?
2. **Structure granularity.** §2 lists six structure labels. Is that the right
   fixed set, or should it be open-ended (risking the §3.3 #5 collision again)?
3. **Cross-recipe identity** (carried from TASK-220 §12). Two recipes creating
   `:Employee` nodes under different identity rules diverge. The §4 scheme makes
   identity deterministic *per recipe*; a graph-level guarantee that two recipes
   agree is not yet specified. Does the model need an identity registry?
