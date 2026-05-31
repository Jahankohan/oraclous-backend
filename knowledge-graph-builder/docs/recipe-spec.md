# Ingestion Recipe Specification

**Status:** v0.2 — TASK-220 (STORY-034). SA-reviewed 2026-05-19; free-text
resolved (position B). Pending finalization (JSON Schema + examples + tests).
**Implements:** ADR-022 (concern-driven data ingestion via agent-authored recipes).

---

## 1. What a recipe is

An **ingestion recipe** is a declarative, reusable specification of how a given
*data shape*, under a given *concern*, projects into knowledge-graph nodes and
edges. Per ADR-022:

- A recipe is **data, not code**. It is a JSON document, validated by a JSON
  Schema. It is *interpreted* by the recipe execution engine — never executed as
  code.
- A recipe is authored **once, at design time**, by the `data-specialist` agent
  reasoning over a *sample* of a source plus a natural-language concern.
- A recipe executes **mechanically, at run time**, over the full dataset — no
  LLM, no agent, deterministic and idempotent.

A new concern is a new recipe. No platform code is written per concern.

## 2. Lifecycle

```
author (design time, agent, over a sample)
  → recipe.status = "draft"
  → human review
  → promote
  → recipe.status = "promoted", version bumped
  → execute (run time, mechanical, over the full source)  ← repeatable
```

Only a `promoted` recipe is executed in production. Drafts are reviewable
artifacts.

## 3. The recipe document

A recipe is one JSON object:

```jsonc
{
  "recipe_format_version": "0.1",     // version of THIS spec/schema
  "id": "rcp_<uuid>",
  "version": 1,                        // version of this recipe; bumped on promote
  "status": "draft",                   // draft | promoted
  "concern": "Understand the reporting structure and team composition.",
  "applies_to": {
    "source_type": "relational",       // relational | csv | json | text | code | ...
    "shape_signature": "<hash/descriptor>"  // how the library matches a source — TASK-224
  },
  "mappings": [ /* §5 — the projection rules */ ],
  "defaults": {
    "provenance": "EXTRACTED",         // §8
    "materialize_fine_grain": false    // §7
  },
  "authoring": {
    "authored_by": "data-specialist",
    "created": "2026-05-19",
    "sample_basis": "first 5 rows of each table; full schema",
    "design_notes_ref": "<link to the authoring notes>"
  }
}
```

`recipe_format_version` versions the **spec**; `version` versions the **recipe**.
The two are independent.

## 4. The data-shape match (`applies_to`)

`applies_to` is what the recipe library (TASK-224) uses to decide whether an
incoming source can reuse this recipe. It carries the `source_type` and a
`shape_signature` — a deterministic descriptor of the source's structure (e.g.
the set of table names + column names + types for a relational source; the JSON
schema for a JSON source). Two sources with the same signature + the same
concern reuse the same recipe with no agent involvement. The exact signature
algorithm is TASK-224's to finalize; the recipe only stores the value.

## 5. Mapping rules

`mappings` is an ordered list of rules. Each rule **matches** a class of
structural unit emitted by a primitive (TASK-222) and declares how it
**projects**. A structural unit has a `kind` (`source`, `table`, `column`,
`row`, `record`, `field`, `section`, `chunk`, `symbol`, …), a path/id, a type,
and sample values.

The set of `unit_kind` values and their attributes is the **structural-unit
vocabulary** — a contract shared with the primitive interface (TASK-222).
TASK-220 owns and publishes this vocabulary; TASK-222's primitives must emit
units that conform to it. The vocabulary is finalized alongside the JSON Schema.

```jsonc
{
  "match": { "unit_kind": "table", "name": "employees" },
  "project_to": "node",              // node | edge | property | skip
  ...                                 // shape depends on project_to
}
```

### 5.1 `project_to: "node"`

```jsonc
{
  "match": { "unit_kind": "table", "name": "employees" },
  "project_to": "node",
  "label": "Employee",                // domain label; reserved namespace per ADR-015
  "identity": { /* §6 */ },
  "materialize": true,                // §7
  "properties": [
    { "name": "name",  "value_from": "column:name" },
    { "name": "title", "value_from": "column:title" }
  ],
  "provenance": "EXTRACTED"           // optional per-rule override of defaults
}
```

### 5.2 `project_to: "edge"`

```jsonc
{
  "match": { "unit_kind": "column", "name": "manager_id", "role": "foreign_key" },
  "project_to": "edge",
  "type": "REPORTS_TO",
  "from": { "node_rule": "employees" },        // the row's own node
  "to":   { "node_rule": "employees",
            "resolve_by": "fk_target" },        // the FK-referenced row's node
  "properties": [],
  "provenance": "EXTRACTED"
}
```

### 5.3 `project_to: "property"`

The unit becomes a property on an already-mapped node rather than its own node.

```jsonc
{ "match": { "unit_kind": "column", "name": "title" },
  "project_to": "property", "on": "employees", "name": "title" }
```

### 5.4 `project_to: "skip"`

The unit is intentionally dropped — noise for *this* concern. Skipping is
explicit and recorded, so a reviewer sees what the recipe chose to ignore.

```jsonc
{ "match": { "unit_kind": "table", "name": "salary_history" },
  "project_to": "skip", "reason": "out of scope for an org-chart concern" }
```

### 5.5 Identifier safety

Node labels, relationship types, and property keys **cannot be parameterized**
in Cypher — the execution engine must build them into the query string. Because
a recipe is agent-authored data, every recipe-supplied label, relationship type,
and property key must, before the engine uses it:

1. **Pass safe-identifier validation** — a strict `[A-Za-z_][A-Za-z0-9_]*`
   allowlist (the existing `structured_ingest_service` / `schema_mapper`
   pattern). An identifier that fails is a *rejected recipe*, not a sanitised
   one. This closes Cypher injection via a malicious identifier.
2. **Be rejected if it intrudes on the ADR-015 reserved namespace**
   (`__Platform__`, `__Entity__`, `__KGBuilder__`, `__Rebac__`, `__System__`)
   or shadows a platform-managed label. A recipe declares *domain* labels only.

A recipe failing either check is invalid — the engine refuses it, it does not
coerce it.

### 5.6 Rule precedence

`mappings` is ordered. A structural unit's **projection** is decided by the
first rule whose `match` matches it — first-match-wins. A unit may still be
*referenced* by other rules (an edge rule's `from` / `to` / `resolve_by`
pointing at a column); referencing is not a second projection and does not
conflict with first-match-wins.

## 6. Identity

Every `node` rule declares how the node's identity is derived. Per ADR-022 D7,
identity is **decoupled from storage primary keys** — so the same real-world
entity converges across rows, columns, and sources.

```jsonc
"identity": {
  "scheme": "deterministic",
  "from": ["column:name"],            // the unit field(s) identity derives from
  "normalize": ["casefold", "trim", "collapse_whitespace"]
}
```

The execution engine `MERGE`s on `(graph_id, label, normalized identity)`. A
recipe may choose a storage key as the identity basis when it genuinely is the
entity's natural key — but that is the recipe's explicit choice, not a default.

## 7. Materialization grain

Per ADR-022 D7: coarse-grain structure (source, table, file, sheet, dataset) is
materialized as nodes by default. Fine-grain units (rows, cells, observations,
symbols) are materialized as nodes **only when the rule sets `materialize:
true`** — otherwise they are properties or are skipped. `defaults.materialize_fine_grain`
sets the baseline; a rule overrides it. This bounds graph size: a million-row
table does not become a million nodes unless the concern demands it.

## 8. Provenance

Every node and edge the engine writes is stamped with a provenance tag, per the
recipe's rules:

- `EXTRACTED` — the element is explicit in the source (a table, a row, an FK).
  Deterministic structured mappings are `EXTRACTED`.
- `INFERRED` — the element is a reasonable inference, not literally in the
  source.

Tags are produced by **recipe rules**, deterministically — not self-reported by
a live model (ADR-022 D8). Genuinely ambiguous mappings are not written by a
silent default; they are surfaced to the agent at authoring time.

## 9. Free-text extraction — `project_to: "text_extraction"`

**Resolved 2026-05-19 — position B.** Sections 5–8 cover *structured* projection
(tables, columns, keys, records, fields); that path is deterministic and
injection-immune. Free-text content — a `raw` evidence field, a prose column, a
document section — does not reduce to a deterministic rule. A recipe extracts
from it by invoking a named **text-extraction primitive**, which **may be
LLM-backed**. ADR-022 D8 is amended accordingly (Amendment 2).

The recipe stays declarative — it *names and configures* the primitive, it does
not reason:

```jsonc
{
  "match": { "unit_kind": "column", "name": "raw", "role": "free_text" },
  "project_to": "text_extraction",
  "primitive": "llm_ner",                       // a registered text-extraction primitive
  "params": { "entity_types": ["Operator", "System", "Standard"] },
  "link_to": { "node_rule": "<row rule>", "edge_type": "MENTIONS" },
  "provenance": "INFERRED"                       // mandatory for this rule kind
}
```

Run-time behaviour and its boundary:

- The named primitive processes the cell text and **returns extracted entities
  and relations as data**. It does **not** write to the graph itself.
- The **deterministic engine** writes those results — under `graph_id` scope,
  the §5.5 identifier-safety checks, and the recipe's rules — and links them to
  the row's node via `link_to`.
- Output is tagged `INFERRED` (never `EXTRACTED`), with a confidence.
- **Trust boundary.** An LLM-backed primitive puts a model on the run-time path,
  so this path is *not* injection-immune (ADR-022 D8 Amendment 2). A
  prompt-injected primitive can yield wrong entities — but, because it only
  *returns data* and the engine does the writing, it cannot escape the tenant
  graph, emit arbitrary Cypher, or touch reserved labels. The blast radius is
  bad `INFERRED` data in one tenant's graph, surfaced by its confidence and tag.

The structured rule kinds (§5.1–§5.6) remain fully deterministic and
injection-immune.

## 10. Versioning

- `recipe_format_version` — bumped when this spec changes shape. The execution
  engine declares which format versions it supports.
- `version` (per recipe) — an integer, bumped each time a reviewed draft is
  promoted. The recipe library (TASK-224) keeps every version; promotion never
  overwrites silently.

## 11. Worked example — relational, org-chart concern

Source: a relational DB with `employees(id, name, title, manager_id→employees,
department_id→departments)`, `departments(id, name)`, `salary_history(...)`.
Concern: *"Understand the reporting structure and team composition."*

| Structural unit | Rule | Result |
|---|---|---|
| `table:employees` | node `Employee`, identity from normalized `name` | one node per employee |
| `column:employees.title` | property `title` on `Employee` | attribute, not a node |
| `column:employees.manager_id` (FK) | edge `REPORTS_TO` (employee → manager's employee) | the reporting tree |
| `column:employees.department_id` (FK) | edge `MEMBER_OF` (employee → department) | team membership |
| `table:departments` | node `Department`, identity from normalized `name` | one node per department |
| `table:salary_history` | `skip` — out of scope for this concern | dropped |

The same source under a *fraud* concern would be a **different recipe** —
`salary_history` would not be skipped, and edges would follow money flow, not
the org tree. Same data, same primitives, different recipe, zero code.

## 12. Open questions

The `/sa` review of this draft (2026-05-19) returned **hold** — five findings.
Findings 2–5 were folded into this revision (§5, §5.5, §5.6); finding 1 is now
resolved. None remain blocking.

1. **§9 — free-text extraction — RESOLVED 2026-05-19.** Position B: a recipe may
   invoke a named, possibly LLM-backed text-extraction primitive. ADR-022 D8 is
   amended (Amendment 2). See §9.
2. **Cross-recipe identity collision.** Two recipes creating `Employee` nodes
   under different identity schemes diverge in one graph. This is resolved by
   the unified model (TASK-221), a graph-level decision. The recipe *format* is
   stable regardless of how TASK-221 resolves it — the format expresses
   per-recipe identity (§6); TASK-221 decides global reconciliation — so
   finalizing TASK-220 is not blocked on TASK-221.

## Deliverables remaining for TASK-220 (post-`/sa`)

- The JSON Schema formalizing §3–§10 once §9 is resolved.
- 2–3 worked example recipes that validate against the schema.
- Schema-validation tests (accept the examples, reject malformed recipes).
- A wiki concept page for the recipe format.
