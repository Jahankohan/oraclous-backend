// =============================================================================
// 2026-05-11 — Assessment substrate schema (STORY-026, TASK-067)
// =============================================================================
//
// Lands the Neo4j schema for the assessment substrate:
//   - 11 node-label uniqueness constraints
//   - Composite + single-property indexes for the documented read patterns
//   - Bootstraps the catalog graph anchor (`__assessments_catalog__`) with the
//     ReBAC reserved-marker convention from ADR-015
//
// Rules honored:
//   - Every statement uses `IF NOT EXISTS` so re-running is idempotent
//   - All assessment-platform nodes carry both the primary label
//     (e.g. `:AssessmentRun`) and the reserved `:__Platform__` marker per
//     ADR-015; that requirement applies to the application-layer writers in
//     TASK-068. This migration declares constraints on the primary label only.
//   - The catalog graph anchor uses `:Graph:__Rebac__ {namespace: '__system__'}`
//     matching the existing pattern in `services/rebac_service.py` and
//     `services/graph_node_service.py`. The application-layer scope enforcer
//     treats `__assessments_catalog__` as a cross-tenant-visible namespace.
//
// Runner: split this file on bare `;` line endings, execute each statement
// through `neo4j_client.execute_write_query`. See
// `app/db/assessment_schema_init.py` for the Python entrypoint.
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Uniqueness constraints (ADR-018 + STORY-026 §Graph Schema + ADR-019)
// -----------------------------------------------------------------------------

CREATE CONSTRAINT assessment_template_id_unique IF NOT EXISTS
FOR (t:AssessmentTemplate) REQUIRE t.template_id IS UNIQUE;

CREATE CONSTRAINT module_id_unique IF NOT EXISTS
FOR (m:Module) REQUIRE m.module_id IS UNIQUE;

CREATE CONSTRAINT subject_id_unique IF NOT EXISTS
FOR (s:Subject) REQUIRE s.subject_id IS UNIQUE;

CREATE CONSTRAINT assessment_run_id_unique IF NOT EXISTS
FOR (r:AssessmentRun) REQUIRE r.run_id IS UNIQUE;

CREATE CONSTRAINT module_run_id_unique IF NOT EXISTS
FOR (mr:ModuleRun) REQUIRE mr.module_run_id IS UNIQUE;

CREATE CONSTRAINT finding_id_unique IF NOT EXISTS
FOR (f:Finding) REQUIRE f.finding_id IS UNIQUE;

CREATE CONSTRAINT source_id_unique IF NOT EXISTS
FOR (s:Source) REQUIRE s.source_id IS UNIQUE;

CREATE CONSTRAINT conflict_id_unique IF NOT EXISTS
FOR (c:Conflict) REQUIRE c.conflict_id IS UNIQUE;

CREATE CONSTRAINT deliverable_id_unique IF NOT EXISTS
FOR (d:Deliverable) REQUIRE d.deliverable_id IS UNIQUE;

CREATE CONSTRAINT unresolved_question_id_unique IF NOT EXISTS
FOR (q:UnresolvedQuestion) REQUIRE q.question_id IS UNIQUE;

CREATE CONSTRAINT registry_item_id_unique IF NOT EXISTS
FOR (r:RegistryItem) REQUIRE r.item_id IS UNIQUE;

// -----------------------------------------------------------------------------
// 2. Indexes for documented read patterns (STORY-026 §Frontend Monitoring
//    Surface + ADR-019 §Read API filtering)
// -----------------------------------------------------------------------------

// `:AssessmentRun` — runs list (filterable by status), per-tenant scoped via
// graph_id. Composite index supports both `WHERE graph_id = $g` and the
// common `(graph_id, status)` filter pattern.
CREATE INDEX assessment_run_graph_status_idx IF NOT EXISTS
FOR (r:AssessmentRun) ON (r.graph_id, r.status);

// `:ModuleRun` — used by the orchestrator's `get_wave_status` query and the
// per-run module-runs UI. Two composite indexes — one for status fan-out, one
// for wave-boundary lookups.
CREATE INDEX module_run_run_status_idx IF NOT EXISTS
FOR (mr:ModuleRun) ON (mr.run_id, mr.status);

CREATE INDEX module_run_run_wave_idx IF NOT EXISTS
FOR (mr:ModuleRun) ON (mr.run_id, mr.wave);

// `:Finding` — primary read access is by run; a finer-grained composite
// supports the findings-table filters (label, confidence) from the
// frontend monitoring surface.
CREATE INDEX finding_run_idx IF NOT EXISTS
FOR (f:Finding) ON (f.run_id);

CREATE INDEX finding_run_label_confidence_idx IF NOT EXISTS
FOR (f:Finding) ON (f.run_id, f.label, f.confidence);

// `:Source` — deduplication and the admin `findings:search?source_url=…`
// path use the normalized URL as the lookup key.
CREATE INDEX source_url_normalized_idx IF NOT EXISTS
FOR (s:Source) ON (s.url_normalized);

// `:UnresolvedQuestion` — orchestrator polls open questions per run to plan
// gap-research wave fan-out (STORY-026 §Coordination Model).
CREATE INDEX unresolved_question_run_status_idx IF NOT EXISTS
FOR (q:UnresolvedQuestion) ON (q.run_id, q.status);

// `:Deliverable` — artifacts list view filters by kind within a run.
CREATE INDEX deliverable_run_kind_idx IF NOT EXISTS
FOR (d:Deliverable) ON (d.run_id, d.kind);

// `:RegistryItem` (ADR-019) — two access patterns:
//   1. Resolve `<kind>/<slug>@<version>` to a single item
//   2. List a user's private/public items in the Registry UI
CREATE INDEX registry_item_kind_slug_version_idx IF NOT EXISTS
FOR (ri:RegistryItem) ON (ri.kind, ri.slug, ri.version);

CREATE INDEX registry_item_owner_visibility_idx IF NOT EXISTS
FOR (ri:RegistryItem) ON (ri.owner_user_id, ri.visibility);

// -----------------------------------------------------------------------------
// 3. Catalog graph anchor (ADR-015 + STORY-026 §Acceptance Criteria)
//    Template-layer nodes (:AssessmentTemplate, :Module, :Subject) and the
//    deduplicated :Source nodes live under graph_id = '__assessments_catalog__'.
//    A `:Graph:__Rebac__` anchor must exist so the existing scope enforcer
//    and federation_service can reason about the namespace.
// -----------------------------------------------------------------------------

MERGE (g:Graph:__Rebac__ {graph_id: '__assessments_catalog__', namespace: '__system__'})
ON CREATE SET
    g.name = 'Assessment Catalog',
    g.description = 'Cross-tenant catalog graph for assessment templates, modules, subjects, and deduplicated sources. See STORY-026 / ADR-018.',
    g.status = 'active',
    g.created_at = datetime(),
    g.federatable = false;

// -----------------------------------------------------------------------------
// 4. Registry catalog anchor (ADR-019). `curated` and `public` Registry
//    items live in the registry catalog graph; `private` items live in the
//    owner's tenant graph.
// -----------------------------------------------------------------------------

MERGE (g:Graph:__Rebac__ {graph_id: '__registry__', namespace: '__system__'})
ON CREATE SET
    g.name = 'Oraclous Registry',
    g.description = 'Cross-tenant catalog graph for curated and public Registry items (Skills, Tools, MCP servers, Agents). Private items live in tenant graphs. See ADR-019 / STORY-028.',
    g.status = 'active',
    g.created_at = datetime(),
    g.federatable = false;
