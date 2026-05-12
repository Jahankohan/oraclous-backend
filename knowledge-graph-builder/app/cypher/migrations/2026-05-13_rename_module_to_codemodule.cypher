// =============================================================================
// 2026-05-13 — Rename :Module → :CodeModule for code-parser (STORY-026, TASK-075)
// =============================================================================
//
// Resolves the `:Module` label collision between the code-parser-service
// (Python-code modules) and the assessment substrate (assessment template
// modules), per ADR-015's marker-namespace convention.
//
// Decision (per Reza, 2026-05-13): rename the code-parser side. The
// assessment substrate keeps `:Module` because its REST + MCP surface is the
// user-facing one. The code-parser side becomes `:CodeModule`.
//
// What this migration does
// ------------------------
//
//   1. Adds the `:CodeModule` label to every existing `:Module` node owned by
//      the code-parser (identifiable by the code-parser's natural-key shape:
//      `graph_id` is a tenant graph and `name` is present), then strips the
//      `:Module` label so the assessment substrate's `:Module` rows are not
//      affected (the code-parser never writes the `:__Platform__` marker on
//      its Module nodes; the assessment substrate always does — that is how
//      we tell the two apart).
//   2. Drops the code-parser `module_unique` constraint on `:Module` and
//      re-declares it on `:CodeModule` with the same property keys.
//   3. Drops the code-parser `code_symbol_search` fulltext index that
//      included `:Module` in its label list and re-declares it with
//      `:CodeModule` instead.
//
// Idempotency
// -----------
//
//   * The `SET m:CodeModule REMOVE m:Module` rewrite is naturally idempotent
//     — re-running it is a no-op once every code-parser Module has been
//     promoted, because the MATCH filter looks specifically for nodes that
//     STILL carry `:Module` AND do NOT carry the `:__Platform__` marker
//     (assessment Modules) — i.e., only the code-parser leftovers.
//   * Constraint drops use `IF EXISTS`; constraint creates use `IF NOT
//     EXISTS`. Same for indexes.
//
// Why this is safe for the assessment substrate
// ---------------------------------------------
//
// Every assessment-substrate `:Module` node carries `:__Platform__` per
// ADR-015 (enforced by `seed_assessment_catalog.py` and
// `assessment_service.py`'s writers). The code-parser-service has never
// added `:__Platform__` to its `:Module` nodes (see
// `code_parser_service.py`, write_code_graph_sync, "2. Module nodes"). So
// `MATCH (m:Module) WHERE NOT m:__Platform__` matches the code-parser's
// rows and only the code-parser's rows.
//
// Runner: same splitter as `assessment_schema_init.py` — split on bare `;`
// line endings, execute each statement through
// `neo4j_client.execute_write_query` (or `async_driver.execute_query`).
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Drop the colliding code-parser constraint on `:Module` BEFORE renaming
//    labels (Neo4j rejects the SET if a constraint on the source label is
//    still active for that property combination on the target side).
// -----------------------------------------------------------------------------

DROP CONSTRAINT module_unique IF EXISTS;

// -----------------------------------------------------------------------------
// 2. Drop the colliding fulltext index that referenced `:Module` in its
//    label list (we re-declare it on `:CodeModule` below).
// -----------------------------------------------------------------------------

DROP INDEX code_symbol_search IF EXISTS;

// -----------------------------------------------------------------------------
// 3. Promote every existing code-parser `:Module` node to `:CodeModule`.
//    Assessment-substrate `:Module` nodes carry `:__Platform__`; code-parser
//    `:Module` nodes do not. Filter on that distinction.
// -----------------------------------------------------------------------------

MATCH (m:Module)
WHERE NOT m:__Platform__
SET m:CodeModule
REMOVE m:Module;

// -----------------------------------------------------------------------------
// 4. Re-declare the code-parser uniqueness constraint on `:CodeModule`.
// -----------------------------------------------------------------------------

CREATE CONSTRAINT code_module_unique IF NOT EXISTS
FOR (m:CodeModule) REQUIRE (m.graph_id, m.name) IS UNIQUE;

// -----------------------------------------------------------------------------
// 5. Re-declare the code-symbol fulltext index with `:CodeModule` instead
//    of `:Module`. The other labels (Function, Class, Variable) are
//    unchanged.
// -----------------------------------------------------------------------------

CREATE FULLTEXT INDEX code_symbol_search IF NOT EXISTS
FOR (n:Function|Class|Variable|CodeModule)
ON EACH [n.name, n.qualified_name, n.docstring];
