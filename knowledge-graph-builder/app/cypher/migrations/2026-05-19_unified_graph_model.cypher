// =============================================================================
// 2026-05-19 — Unified Source → Structure → Entity graph model (STORY-034, TASK-221)
// =============================================================================
//
// Establishes the constraints for the unified graph model that recipes
// (TASK-220) project into. Full design: knowledge-graph-builder/docs/
// unified-graph-model.md. Decision record: ADR-022 D7.
//
// Node families
// -------------
//
//   * :Source        — one per ingested data source.
//   * :Table :Sheet :File :Chunk — containers inside a source (a fixed,
//     platform-owned set; recipes never invent these).
//   * :__Entity__     — a real thing; carries one recipe-supplied domain label.
//
// Identity (see docs §4) is decoupled from storage primary keys:
//   * :Source     is unique on (graph_id, source_id)
//   * containers  are unique on (graph_id, id)
//   * :__Entity__ is unique on (graph_id, id)
//
// Idempotency
// -----------
//
//   Every statement uses `IF NOT EXISTS`; re-running this migration is a no-op.
//   No data is migrated — this establishes the model on a clean database
//   (the project is in development; no backward compatibility — see docs §1).
//
//   The composite uniqueness constraints create backing range indexes; a query
//   filtering on the `graph_id` prefix is served by them, so no separate
//   per-label `graph_id` index is declared (avoids blanket-indexing).
//
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Source — the top node for each ingested data source.
// -----------------------------------------------------------------------------

CREATE CONSTRAINT ugm_source_key IF NOT EXISTS
FOR (s:Source) REQUIRE (s.graph_id, s.source_id) IS UNIQUE;

// -----------------------------------------------------------------------------
// 2. Containers — groupings inside a source. Fixed, platform-owned label set.
// -----------------------------------------------------------------------------

CREATE CONSTRAINT ugm_table_key IF NOT EXISTS
FOR (t:Table) REQUIRE (t.graph_id, t.id) IS UNIQUE;

CREATE CONSTRAINT ugm_sheet_key IF NOT EXISTS
FOR (sh:Sheet) REQUIRE (sh.graph_id, sh.id) IS UNIQUE;

CREATE CONSTRAINT ugm_file_key IF NOT EXISTS
FOR (f:File) REQUIRE (f.graph_id, f.id) IS UNIQUE;

CREATE CONSTRAINT ugm_chunk_key IF NOT EXISTS
FOR (c:Chunk) REQUIRE (c.graph_id, c.id) IS UNIQUE;

// -----------------------------------------------------------------------------
// 3. Entity — a real thing. One node per real thing (docs §2); identity is
//    decoupled from storage primary keys (docs §4).
// -----------------------------------------------------------------------------

CREATE CONSTRAINT ugm_entity_key IF NOT EXISTS
FOR (e:__Entity__) REQUIRE (e.graph_id, e.id) IS UNIQUE;
