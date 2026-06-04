---
title: "Phase 5 — Pipeline Templates (HR, Research, Legal, Codebase)"
author: Data Engineer Specialist
date: 2026-06-01
status: draft
source: ORA-344
implements: ~
reviewed-by: ~
implemented-in: ~
---

## Overview

This spec defines a library of four pre-built pipeline templates for the Visual Flow Studio. Each template is a pre-configured pipeline that a user can load onto the canvas, fill in their data source and graph name, and execute — reducing time-to-first-graph from hours to minutes. Templates align with the node types defined in the Visual Flow Studio spec ([ORA-342](/ORA/issues/ORA-342)) and map every node to at least three existing Oraclous API operations.

---

## Requirements

1. Exactly four templates ship in the initial catalogue: HR/People Graph, Research Knowledge Base, Legal Document Graph, and Codebase Architecture Graph.
2. Every template must map each pipeline node to a named Oraclous API operation.
3. Templates are serialised in JSON and validated against the template schema defined in this spec.
4. Three fields are user-configurable on every template: `data_source`, `graph_name`, and `ontology_mode`.
5. Domain-specific ontologies embedded in each template must match the canonical ontology definitions maintained by the Data Engineer Specialist.
6. Templates must be loadable from a static catalogue endpoint; no runtime LLM generation is required.
7. Template instantiation must produce a valid recipe sequence that the backend can execute without manual editing.
8. All node identifiers must use `snake_case`; all relationship types must use `SCREAMING_SNAKE_CASE`.

---

## Template File Format

### Schema

Templates are stored as JSON. The schema is:

```json
{
  "$schema": "https://oraclous.dev/schemas/pipeline-template/v1.json",
  "version": "1.0",
  "id": "<kebab-case-slug>",
  "name": "<Human-readable display name>",
  "domain": "<hr | research | legal | codebase>",
  "description": "<One-sentence description shown in template picker>",
  "thumbnail": "<relative path to a 400x240 PNG preview image>",
  "parameters": [
    {
      "key": "<param_key>",
      "label": "<Human-readable label>",
      "type": "<string | enum | boolean | integer>",
      "required": true,
      "default": "<default value or null>",
      "options": ["<option1>", "<option2>"],
      "hint": "<Help text shown in the Studio parameter panel>"
    }
  ],
  "ontology": {
    "entity_types": [
      {
        "label": "<NodeLabel>",
        "properties": ["<prop1>", "<prop2>"]
      }
    ],
    "relationship_types": [
      {
        "type": "<RELATIONSHIP_TYPE>",
        "from": "<SourceLabel>",
        "to": "<TargetLabel>",
        "properties": ["<prop1>"]
      }
    ]
  },
  "nodes": [
    {
      "id": "<node_id>",
      "type": "<source | transform | sink>",
      "label": "<Display label on canvas>",
      "operation": "<API operation identifier>",
      "config": {
        "<key>": "<value or {{param_key}} substitution>"
      }
    }
  ],
  "edges": [
    {
      "id": "<edge_id>",
      "source": "<node_id>",
      "target": "<node_id>",
      "label": "<optional edge label>"
    }
  ]
}
```

### Field Rules

| Field | Required | Notes |
|---|---|---|
| `version` | yes | Semver string; current version is `"1.0"` |
| `id` | yes | Unique slug; kebab-case; used as URL path segment |
| `domain` | yes | One of: `hr`, `research`, `legal`, `codebase` |
| `parameters` | yes | At minimum: `data_source`, `graph_name`, `ontology_mode` |
| `ontology` | yes | Must match canonical ontology for the domain |
| `nodes` | yes | Minimum 3 nodes; every node must have a valid `operation` |
| `edges` | yes | DAG only; no cycles |
| `thumbnail` | no | Shown in catalogue browser |

### Parameter Substitution

Parameter values are injected into node `config` at instantiation time using `{{param_key}}` placeholders:

```json
"config": {
  "graph_name": "{{graph_name}}",
  "source_path": "{{data_source}}"
}
```

Substitution is a pure string replacement; no expression evaluation.

---

## API Operation Registry

Each node `operation` field must reference one of the following registered Oraclous API operations:

| Operation ID | Method + Path | Purpose |
|---|---|---|
| `graph.create` | `POST /api/v1/graphs` | Create a new knowledge graph |
| `graph.ingest` | `POST /api/v1/graphs/{graph_id}/ingest` | Ingest one or more documents (PDF, DOCX, TXT) |
| `graph.documents.create` | `POST /api/v1/graphs/{graph_id}/documents` | Add a single document node |
| `graph.ingest.incremental` | `POST /api/v1/graphs/{graph_id}/ingest/incremental` | Incremental document ingestion |
| `graph.code_ingest` | `POST /api/v1/graphs/{graph_id}/code-ingest` | Ingest a codebase from a Git repository or local path |
| `graph.code.symbols` | `GET /api/v1/graphs/{graph_id}/code/symbols` | Query extracted code symbols |
| `graph.connector.create` | `POST /api/v1/graphs/{graph_id}/connectors/database` | Register a database connector |
| `graph.connector.sync` | `POST /api/v1/graphs/{graph_id}/connectors/database/{connector_id}/sync` | Trigger a connector sync |
| `graph.communities.build` | `POST /api/v1/graphs/{graph_id}/communities` | Run Leiden community detection |
| `graph.federation.query` | `POST /api/v1/graphs/{graph_id}/federation/query` | Cross-graph federation query |
| `graph.evaluation.run` | `POST /api/v1/graphs/{graph_id}/evaluation` | Run RAGAS evaluation on the graph |

---

## Template Catalogue

### Template 1: HR / People Graph

**ID:** `hr-people-graph`
**Domain:** `hr`
**Description:** Build an org chart and skills knowledge graph from HR exports — CSV, HRIS database, or LDAP directory.

#### Parameters

| Key | Label | Type | Required | Default | Hint |
|---|---|---|---|---|---|
| `graph_name` | Graph name | string | yes | `"HR Knowledge Graph"` | Display name for the created graph |
| `data_source` | Data source path | string | yes | — | Local file path, S3 URI, or DB connection string |
| `source_type` | Source type | enum | yes | `"csv"` | `csv`, `database`, `pdf` |
| `ontology_mode` | Extraction mode | enum | yes | `"guided"` | `guided` uses HR ontology; `free-form` lets the LLM decide types |
| `include_skills` | Include skills | boolean | no | `true` | Extract Skill nodes and HAS_SKILL relationships |
| `include_projects` | Include projects | boolean | no | `false` | Extract Project nodes and WORKS_ON relationships |

#### Ontology

```yaml
entity_types:
  - Person: [name, email, employee_id, department, location]
  - Organization: [name, industry, size, headquarters]
  - Department: [name, cost_center, head_count]
  - Role: [title, level, function]
  - Skill: [name, category, proficiency_levels]
  - Project: [name, status, start_date, end_date]

relationship_types:
  - WORKS_FOR: Person → Organization [position, start_date, end_date, employment_type]
  - REPORTS_TO: Person → Person [since, direct]
  - MANAGES: Person → Person [since]
  - BELONGS_TO: Person → Department [since]
  - HAS_SKILL: Person → Skill [proficiency, certified, years_experience]
  - HOLDS_ROLE: Person → Role [since, end_date]
  - WORKS_ON: Person → Project [role, allocation_pct, since]
```

#### Pipeline Nodes

| Node ID | Type | Label | Operation | Config |
|---|---|---|---|---|
| `create_graph` | source | Create Graph | `graph.create` | `{name: "{{graph_name}}"}` |
| `ingest_source` | source | Ingest HR Data | `graph.ingest` | `{source_path: "{{data_source}}", source_type: "{{source_type}}", ontology_mode: "{{ontology_mode}}"}` |
| `extract_entities` | transform | Extract Entities | `graph.ingest` | Implicit — extraction is part of ingest; ontology injected via `ontology_config` |
| `build_communities` | transform | Detect Clusters | `graph.communities.build` | `{algorithm: "leiden", levels: 2}` |
| `graph_output` | sink | HR Knowledge Graph | — | Result node; no API call |

#### Edges

```
create_graph → ingest_source → extract_entities → build_communities → graph_output
```

#### Mapped API Operations (≥3)

1. `POST /api/v1/graphs` — creates the graph container
2. `POST /api/v1/graphs/{graph_id}/ingest` — ingests HR files with guided ontology
3. `POST /api/v1/graphs/{graph_id}/communities` — runs Leiden to cluster people/teams

#### Sample Query (post-build)

```cypher
// Find all direct reports of a manager
MATCH (mgr:Person {name: $manager_name, graph_id: $graph_id})
      <-[:REPORTS_TO]-(report:Person {graph_id: $graph_id})
RETURN report.name, report.email
ORDER BY report.name
```

---

### Template 2: Research Knowledge Base

**ID:** `research-knowledge-base`
**Domain:** `research`
**Description:** Build a citation and concept graph from academic papers — PDFs, BibTeX files, or arXiv exports.

#### Parameters

| Key | Label | Type | Required | Default | Hint |
|---|---|---|---|---|---|
| `graph_name` | Graph name | string | yes | `"Research Knowledge Base"` | Display name |
| `data_source` | Data source path | string | yes | — | Directory of PDFs or BibTeX file |
| `source_type` | Source type | enum | yes | `"pdf"` | `pdf`, `bibtex`, `arxiv_export` |
| `ontology_mode` | Extraction mode | enum | yes | `"guided"` | `guided` or `free-form` |
| `extract_datasets` | Extract datasets | boolean | no | `true` | Extract Dataset nodes |
| `extract_methods` | Extract methods | boolean | no | `true` | Extract Method nodes |
| `run_evaluation` | Run evaluation | boolean | no | `false` | Run RAGAS evaluation after build |

#### Ontology

```yaml
entity_types:
  - Paper: [title, abstract, year, doi, venue, open_access]
  - Author: [name, affiliation, email, orcid, h_index]
  - Institution: [name, country, type, ror_id]
  - Topic: [name, field, subfield, keywords]
  - Dataset: [name, size, format, url, license]
  - Method: [name, type, description, paper_origin]

relationship_types:
  - AUTHORED: Author → Paper [order, corresponding, affiliation_at_time]
  - CITES: Paper → Paper [context, section, citation_type]
  - AFFILIATED_WITH: Author → Institution [role, period_start, period_end]
  - COVERS_TOPIC: Paper → Topic [primary, weight]
  - USES_METHOD: Paper → Method [for_task, variant]
  - USES_DATASET: Paper → Dataset [for_task, split]
  - INTRODUCES: Paper → Method [claimed_improvement]
  - RELATED_TO: Topic → Topic [relationship_type]
```

#### Pipeline Nodes

| Node ID | Type | Label | Operation | Config |
|---|---|---|---|---|
| `create_graph` | source | Create Graph | `graph.create` | `{name: "{{graph_name}}"}` |
| `ingest_papers` | source | Ingest Papers | `graph.ingest` | `{source_path: "{{data_source}}", source_type: "{{source_type}}", ontology_mode: "{{ontology_mode}}"}` |
| `extract_citations` | transform | Extract Citations | `graph.ingest` | Implicit via ontology; CITES relationships extracted during entity pass |
| `build_communities` | transform | Cluster Topics | `graph.communities.build` | `{algorithm: "leiden", levels: 3}` |
| `evaluate` | transform | Evaluate Quality | `graph.evaluation.run` | `{enabled: "{{run_evaluation}}"}` |
| `graph_output` | sink | Research Graph | — | Result node |

#### Edges

```
create_graph → ingest_papers → extract_citations → build_communities → evaluate → graph_output
```

#### Mapped API Operations (≥3)

1. `POST /api/v1/graphs` — creates the graph
2. `POST /api/v1/graphs/{graph_id}/ingest` — ingests PDFs with research ontology
3. `POST /api/v1/graphs/{graph_id}/communities` — clusters topics and research areas
4. `POST /api/v1/graphs/{graph_id}/evaluation` — optional RAGAS evaluation

#### Sample Query (post-build)

```cypher
// Find papers that cite a given paper with context
MATCH (source:Paper {doi: $doi, graph_id: $graph_id})
      <-[:CITES {graph_id: $graph_id}]-(citing:Paper {graph_id: $graph_id})
RETURN citing.title, citing.year, citing.venue
ORDER BY citing.year DESC
```

---

### Template 3: Legal Document Graph

**ID:** `legal-document-graph`
**Domain:** `legal`
**Description:** Extract clauses, obligations, parties, and cross-references from contracts, regulations, and legal documents.

#### Parameters

| Key | Label | Type | Required | Default | Hint |
|---|---|---|---|---|---|
| `graph_name` | Graph name | string | yes | `"Legal Document Graph"` | Display name |
| `data_source` | Data source path | string | yes | — | Directory of PDFs or DOCX files |
| `source_type` | Source type | enum | yes | `"pdf"` | `pdf`, `docx`, `txt` |
| `document_class` | Document class | enum | yes | `"contract"` | `contract`, `regulation`, `policy`, `court_ruling` |
| `ontology_mode` | Extraction mode | enum | yes | `"guided"` | `guided` strongly recommended for legal precision |
| `chunk_size` | Chunk size (tokens) | integer | no | `1000` | Larger chunks preserve clause context; 800–1200 recommended |

#### Ontology

```yaml
entity_types:
  - Contract: [title, effective_date, jurisdiction, contract_type, status]
  - Regulation: [name, jurisdiction, effective_date, regulator, citation]
  - Clause: [number, title, text_excerpt, category, is_standard]
  - Party: [name, role, jurisdiction, entity_type]
  - Obligation: [description, deadline, penalty, subject_party]
  - Right: [description, beneficiary, conditions]
  - Definition: [term, meaning, source_clause]

relationship_types:
  - CONTAINS: Contract → Clause [section, order]
  - CONTAINS: Regulation → Clause [article, paragraph]
  - PARTY_TO: Party → Contract [role, signatory_date]
  - SUBJECT_TO: Party → Regulation [since, compliance_status]
  - IMPOSES: Clause → Obligation [on_party_role, conditional]
  - GRANTS: Clause → Right [to_party_role, conditional]
  - DEFINES: Clause → Definition [scope]
  - REFERENCES: Clause → Clause [reference_type]
  - SUPERSEDES: Regulation → Regulation [effective_date]
  - AMENDS: Contract → Contract [amendment_date, sections_affected]
```

#### Pipeline Nodes

| Node ID | Type | Label | Operation | Config |
|---|---|---|---|---|
| `create_graph` | source | Create Graph | `graph.create` | `{name: "{{graph_name}}"}` |
| `ingest_documents` | source | Ingest Documents | `graph.ingest` | `{source_path: "{{data_source}}", source_type: "{{source_type}}", chunk_size: "{{chunk_size}}", ontology_mode: "{{ontology_mode}}"}` |
| `extract_clauses` | transform | Extract Clauses | `graph.ingest` | Implicit — guided ontology extraction produces Clause and Obligation nodes |
| `link_references` | transform | Link Cross-References | `graph.ingest.incremental` | Second-pass extraction to resolve REFERENCES and SUPERSEDES edges |
| `graph_output` | sink | Legal Graph | — | Result node |

#### Edges

```
create_graph → ingest_documents → extract_clauses → link_references → graph_output
```

#### Mapped API Operations (≥3)

1. `POST /api/v1/graphs` — creates the graph
2. `POST /api/v1/graphs/{graph_id}/ingest` — first-pass extraction with guided legal ontology
3. `POST /api/v1/graphs/{graph_id}/ingest/incremental` — second-pass cross-reference linking
4. `POST /api/v1/graphs/{graph_id}/documents` — individual document registration (optional for batch tracking)

#### Sample Query (post-build)

```cypher
// Find all obligations imposed on a party by a specific contract
MATCH (c:Contract {title: $contract_title, graph_id: $graph_id})
      -[:CONTAINS]->(clause:Clause {graph_id: $graph_id})
      -[:IMPOSES]->(obl:Obligation {graph_id: $graph_id})
WHERE obl.subject_party = $party_name
RETURN clause.number, clause.title, obl.description, obl.deadline
ORDER BY clause.number
```

---

### Template 4: Codebase Architecture Graph

**ID:** `codebase-architecture-graph`
**Domain:** `codebase`
**Description:** Map modules, classes, functions, call graphs, and dependencies from a Git repository.

#### Parameters

| Key | Label | Type | Required | Default | Hint |
|---|---|---|---|---|---|
| `graph_name` | Graph name | string | yes | `"Codebase Architecture"` | Display name |
| `data_source` | Repository path or URL | string | yes | — | Local path or HTTPS Git URL |
| `language` | Primary language | enum | yes | `"python"` | `python`, `typescript`, `javascript`, `java`, `go` |
| `ontology_mode` | Extraction mode | enum | yes | `"guided"` | Always `guided` for code; `free-form` not applicable |
| `depth` | Extraction depth | enum | yes | `"function"` | `file` (fast), `function` (recommended), `statement` (detailed) |
| `include_tests` | Include test files | boolean | no | `true` | Extract Test nodes and TESTS relationships |
| `include_dependencies` | Include dependencies | boolean | no | `true` | Extract Dependency nodes from package manifests |

#### Ontology

```yaml
entity_types:
  - Repository: [name, url, language, commit_sha, branch]
  - Module: [name, path, language, line_count]
  - Class: [name, module, visibility, is_abstract, line_number]
  - Function: [name, module, class, visibility, parameters, return_type, line_number, docstring]
  - Variable: [name, type, scope, module, class]
  - Test: [name, type, module, covers_function]
  - Dependency: [name, version, type, resolved_version]

relationship_types:
  - CONTAINS: Repository → Module [path]
  - IMPORTS: Module → Module [alias, import_type]
  - DEFINES: Module → Class [line_number]
  - DEFINES: Module → Function [line_number]
  - HAS_METHOD: Class → Function [visibility, is_static, is_classmethod]
  - CALLS: Function → Function [call_site_line, call_type]
  - INHERITS: Class → Class [inheritance_type]
  - INSTANTIATES: Function → Class [call_site_line]
  - DEPENDS_ON: Module → Dependency [version_constraint, import_scope]
  - TESTS: Test → Function [coverage_type, assertion_count]
  - TESTS: Test → Class [coverage_type]
```

#### Pipeline Nodes

| Node ID | Type | Label | Operation | Config |
|---|---|---|---|---|
| `create_graph` | source | Create Graph | `graph.create` | `{name: "{{graph_name}}"}` |
| `ingest_code` | source | Ingest Repository | `graph.code_ingest` | `{repo_path: "{{data_source}}", language: "{{language}}", depth: "{{depth}}", include_tests: "{{include_tests}}", include_dependencies: "{{include_dependencies}}"}` |
| `query_symbols` | transform | Verify Symbols | `graph.code.symbols` | `{limit: 100}` — verification step, not required for pipeline execution |
| `build_communities` | transform | Cluster Modules | `graph.communities.build` | `{algorithm: "leiden", levels: 2}` |
| `graph_output` | sink | Codebase Graph | — | Result node |

#### Edges

```
create_graph → ingest_code → query_symbols → build_communities → graph_output
```

#### Mapped API Operations (≥3)

1. `POST /api/v1/graphs` — creates the graph
2. `POST /api/v1/graphs/{graph_id}/code-ingest` — runs Tree-sitter AST extraction
3. `GET /api/v1/graphs/{graph_id}/code/symbols` — queries extracted symbols (verification)
4. `POST /api/v1/graphs/{graph_id}/communities` — clusters modules into architectural layers

#### Sample Query (post-build)

```cypher
// Find all callers of a function
MATCH (caller:Function {graph_id: $graph_id})
      -[:CALLS]->(target:Function {name: $function_name, graph_id: $graph_id})
RETURN caller.name, caller.module, caller.line_number
ORDER BY caller.module, caller.line_number
```

---

## Template Loading Spec

### Catalogue Endpoint

The backend exposes a read-only catalogue. No new backend work is required for MVP — templates can be served as static JSON from the frontend bundle. A backend route is the preferred approach for v1.1+ to support user-defined template sharing.

**Proposed routes (v1.1):**

```
GET  /api/v1/pipeline-templates                       — list all templates
GET  /api/v1/pipeline-templates/{template_id}         — get single template
POST /api/v1/pipeline-templates/{template_id}/instantiate  — instantiate with params
```

**Instantiate request body:**

```json
{
  "parameters": {
    "graph_name": "My HR Graph",
    "data_source": "/uploads/hr-export.csv",
    "source_type": "csv",
    "ontology_mode": "guided"
  }
}
```

**Instantiate response:**

```json
{
  "template_id": "hr-people-graph",
  "instance_id": "<uuid>",
  "resolved_nodes": [...],
  "resolved_edges": [...],
  "execution_recipe": [...]
}
```

### Loading into Visual Flow Studio Canvas

When a user selects a template in the studio:

1. Frontend fetches `GET /api/v1/pipeline-templates/{template_id}` (or reads from bundled static JSON for MVP).
2. A parameter panel slides open, showing all `parameters` with `label`, `hint`, and `default`.
3. User fills required fields; optional fields are pre-populated with defaults.
4. On confirm, frontend calls `POST /api/v1/pipeline-templates/{template_id}/instantiate` (or performs local substitution for MVP).
5. The resolved `nodes` and `edges` arrays are passed to the React Flow `setNodes` / `setEdges` calls.
6. Canvas renders the full pipeline; user can edit any node before running.

**Node → React Flow mapping:**

| Template node `type` | React Flow node type | Visual style |
|---|---|---|
| `source` | `sourceNode` | Green header, input connector only on right |
| `transform` | `transformNode` | Blue header, input left / output right |
| `sink` | `sinkNode` | Orange header, input connector only on left |

---

## Parameterization Model

### Universal Parameters (all templates)

| Key | Type | Required | Notes |
|---|---|---|---|
| `graph_name` | string | yes | Used as the `name` field in `POST /api/v1/graphs` |
| `data_source` | string | yes | File path, directory, S3 URI, or DB connection string |
| `ontology_mode` | enum | yes | `guided` \| `free-form` \| `hybrid`; passed to ingest as `ontology_config.mode` |

### Domain-Specific Parameters

Each template adds 3–4 domain-specific parameters on top of the universal set. All are documented in the template's **Parameters** section above.

### Ontology Injection

When `ontology_mode = "guided"`, the template's embedded `ontology` block is serialised and passed to the ingest endpoint as the `ontology_config` payload:

```json
{
  "source_path": "/data/hr-export.csv",
  "ontology_config": {
    "mode": "guided",
    "entity_types": ["Person", "Organization", "Department", "Role", "Skill"],
    "relationship_types": ["WORKS_FOR", "REPORTS_TO", "HAS_SKILL", "MANAGES"]
  }
}
```

When `ontology_mode = "free-form"`, `ontology_config` is omitted and the LLM determines entity types.

When `ontology_mode = "hybrid"`, the `ontology_config` is passed as `mode: "hint"` — the LLM can extend the provided types but not contradict them.

---

## Migration Plan

No migration is required. Pipeline templates are additive — they do not modify the existing graph schema or API surface. The `instantiate` endpoint is a new thin route that performs parameter substitution and returns a resolved pipeline. Existing graphs are unaffected.

For MVP, templates can ship as static JSON files in the frontend bundle (`src/templates/`) with local substitution in the Studio component — zero backend changes required.

---

## Test Criteria

1. **Schema validation:** each of the four template JSON files parses without error against the template schema; `required` fields are present; no unregistered `operation` identifiers appear.
2. **Parameter substitution:** calling `instantiate` with all required parameters returns a resolved payload where no `{{...}}` placeholders remain.
3. **Node-to-API mapping:** every node `operation` in all four templates resolves to an entry in the API operation registry table; `GET /api/v1/pipeline-templates/{id}` returns HTTP 200 with valid JSON.
4. **Canvas rendering:** loading any template into the Visual Flow Studio canvas renders without JavaScript errors; all nodes and edges appear at expected positions.
5. **Ontology injection:** with `ontology_mode = "guided"`, the resolved ingest node config includes a non-empty `ontology_config`; with `ontology_mode = "free-form"`, `ontology_config` is absent.
6. **End-to-end smoke (HR template):** using test fixture `tests/fixtures/hr-sample.csv`, executing the instantiated HR pipeline creates a graph with at least one `Person` node and one `REPORTS_TO` relationship.
7. **End-to-end smoke (Codebase template):** pointing the codebase template at the `oraclous-data-studio` repo itself creates a graph with `Module` and `Function` nodes and `CALLS` relationships.

---

## Performance Expectations

| Template | Expected nodes | Expected edges | Expected build time | P95 query latency |
|---|---|---|---|---|
| HR / People Graph (1k employees) | ~5 000 | ~15 000 | < 3 min | < 200 ms |
| Research KB (500 papers) | ~10 000 | ~30 000 | < 5 min | < 300 ms |
| Legal Document Graph (100 contracts) | ~8 000 | ~20 000 | < 4 min | < 250 ms |
| Codebase Architecture (50k LOC) | ~3 000 | ~12 000 | < 2 min | < 150 ms |

All estimates assume `claude-haiku-4-5` for extraction, standard Neo4j AuraDB tier, and chunking at 512 tokens with 64-token overlap.
