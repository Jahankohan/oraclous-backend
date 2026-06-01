"""Pydantic schemas and built-in catalogue for Pipeline Templates (ORA-352).

Templates are PipelineDefinition skeletons with {{param_key}} placeholders in
config fields.  The frontend substitutes param values before loading nodes/edges
onto the React Flow canvas.

Template JSON shape mirrors PipelineDefinition from the Visual Flow Studio spec
(knowledge-base/specs/ui/visual-flow-studio.md).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Template parameter descriptor
# ---------------------------------------------------------------------------


class PipelineTemplateParam(BaseModel):
    key: str = Field(
        ..., description="Placeholder key without braces, e.g. 'graph_name'"
    )
    label: str = Field(
        ..., description="Human-readable label shown in substitution dialog"
    )
    description: str = Field("", description="Helper text shown below the field")
    type: Literal["text", "email", "url", "select", "file"] = "text"
    required: bool = True
    default: str | None = None
    options: list[str] | None = Field(
        None, description="Allowed values (only meaningful when type='select')"
    )


# ---------------------------------------------------------------------------
# Pipeline template metadata + skeleton
# ---------------------------------------------------------------------------


class PipelineTemplateNode(BaseModel):
    id: str
    type: str
    position: dict[str, float]
    data: dict[str, Any]


class PipelineTemplateEdge(BaseModel):
    id: str
    source: str
    target: str
    type: Literal["data-flow", "trigger"] = "data-flow"


class PipelineTemplate(BaseModel):
    id: str = Field(..., description="Stable slug used as catalogue key")
    name: str
    description: str
    domain: Literal["hr", "research", "legal", "codebase"]
    tags: list[str] = Field(default_factory=list)
    params: list[PipelineTemplateParam]
    nodes: list[PipelineTemplateNode]
    edges: list[PipelineTemplateEdge]


# ---------------------------------------------------------------------------
# Catalogue response
# ---------------------------------------------------------------------------


class PipelineTemplateSummary(BaseModel):
    """Lightweight catalogue entry — no nodes/edges, only metadata."""

    id: str
    name: str
    description: str
    domain: str
    tags: list[str]
    node_count: int
    param_count: int


class PipelineTemplateCatalogueResponse(BaseModel):
    templates: list[PipelineTemplateSummary]
    total: int


# ---------------------------------------------------------------------------
# Built-in template catalogue
# ---------------------------------------------------------------------------

_HR_TEMPLATE = PipelineTemplate(
    id="hr-people-graph",
    name="HR / People Graph",
    description=(
        "Ingest HR records (CSV or spreadsheet) and build a people-and-org knowledge graph. "
        "Extracts Person, Role, Team, and Project entities; deduplicates employees; "
        "detects org communities; and summarises team clusters."
    ),
    domain="hr",
    tags=["hr", "people", "org-chart", "starter"],
    params=[
        PipelineTemplateParam(
            key="graph_name",
            label="Graph Name",
            description="Name for the new knowledge graph",
            type="text",
            required=True,
        ),
        PipelineTemplateParam(
            key="llm_config_id",
            label="LLM Configuration",
            description="Select a saved LLM config to drive entity extraction",
            type="select",
            required=True,
        ),
        PipelineTemplateParam(
            key="reviewer_email",
            label="Reviewer E-mail",
            description="Who receives the human-review notification before the graph finalises",
            type="email",
            required=False,
            default="",
        ),
    ],
    nodes=[
        PipelineTemplateNode(
            id="n-create-graph",
            type="create-graph",
            position={"x": 60, "y": 240},
            data={
                "label": "Create Graph",
                "config": {
                    "name": "{{graph_name}}",
                    "description": "HR people and organisation graph",
                    "llm_config_id": "{{llm_config_id}}",
                    "ontology_template": "hr",
                },
            },
        ),
        PipelineTemplateNode(
            id="n-ingest-file",
            type="ingest-file",
            position={"x": 320, "y": 120},
            data={
                "label": "Ingest HR Records",
                "config": {
                    "mode": "full",
                },
            },
        ),
        PipelineTemplateNode(
            id="n-ontology-apply",
            type="ontology-apply",
            position={"x": 580, "y": 120},
            data={
                "label": "Apply HR Ontology",
                "config": {
                    "entity_types": [
                        "Person",
                        "Organization",
                        "Role",
                        "Skill",
                        "Project",
                    ],
                    "relationship_types": [
                        "WORKS_FOR",
                        "HAS_SKILL",
                        "MANAGES",
                        "WORKS_ON",
                        "REPORTS_TO",
                    ],
                },
            },
        ),
        PipelineTemplateNode(
            id="n-entity-dedup",
            type="entity-dedup",
            position={"x": 840, "y": 120},
            data={
                "label": "Deduplicate People",
                "config": {
                    "strategy": "hybrid",
                    "threshold": 0.92,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-community-detect",
            type="community-detect",
            position={"x": 1100, "y": 120},
            data={
                "label": "Detect Org Communities",
                "config": {
                    "algorithm": "leiden",
                    "resolution": 1.0,
                    "min_community_size": 3,
                    "max_levels": 3,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-community-summarize",
            type="community-summarize",
            position={"x": 1360, "y": 120},
            data={
                "label": "Summarise Teams",
                "config": {
                    "level": 0,
                    "max_tokens": 512,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-graph-output",
            type="graph-output",
            position={"x": 1620, "y": 240},
            data={
                "label": "HR Knowledge Graph",
                "config": {
                    "graph_id": "from create-graph node",
                },
            },
        ),
    ],
    edges=[
        PipelineTemplateEdge(
            id="e-cg-if", source="n-create-graph", target="n-ingest-file"
        ),
        PipelineTemplateEdge(
            id="e-if-oa", source="n-ingest-file", target="n-ontology-apply"
        ),
        PipelineTemplateEdge(
            id="e-oa-ed", source="n-ontology-apply", target="n-entity-dedup"
        ),
        PipelineTemplateEdge(
            id="e-ed-cd", source="n-entity-dedup", target="n-community-detect"
        ),
        PipelineTemplateEdge(
            id="e-cd-cs", source="n-community-detect", target="n-community-summarize"
        ),
        PipelineTemplateEdge(
            id="e-cs-go", source="n-community-summarize", target="n-graph-output"
        ),
    ],
)

_RESEARCH_TEMPLATE = PipelineTemplate(
    id="research-knowledge-base",
    name="Research Knowledge Base",
    description=(
        "Turn academic papers, reports, or documentation URLs into a structured research graph. "
        "Extracts Author, Paper, Topic, Institution, Method, and Dataset entities; "
        "builds a citation-aware similarity index; and surfaces research communities."
    ),
    domain="research",
    tags=["research", "academic", "papers", "citations"],
    params=[
        PipelineTemplateParam(
            key="graph_name",
            label="Graph Name",
            description="Name for the research knowledge graph",
            type="text",
            required=True,
        ),
        PipelineTemplateParam(
            key="llm_config_id",
            label="LLM Configuration",
            description="LLM config used for entity extraction from papers",
            type="select",
            required=True,
        ),
        PipelineTemplateParam(
            key="source_url",
            label="Seed URL",
            description="Optional starting URL (e.g. an arXiv search or a docs page)",
            type="url",
            required=False,
            default="",
        ),
    ],
    nodes=[
        PipelineTemplateNode(
            id="n-create-graph",
            type="create-graph",
            position={"x": 60, "y": 300},
            data={
                "label": "Create Graph",
                "config": {
                    "name": "{{graph_name}}",
                    "description": "Research and academic knowledge graph",
                    "llm_config_id": "{{llm_config_id}}",
                    "ontology_template": "research",
                },
            },
        ),
        PipelineTemplateNode(
            id="n-ingest-url",
            type="ingest-url",
            position={"x": 320, "y": 120},
            data={
                "label": "Ingest from URL",
                "config": {
                    "url": "{{source_url}}",
                    "mode": "full",
                },
            },
        ),
        PipelineTemplateNode(
            id="n-ingest-file",
            type="ingest-file",
            position={"x": 320, "y": 480},
            data={
                "label": "Ingest PDF Papers",
                "config": {
                    "mode": "full",
                },
            },
        ),
        PipelineTemplateNode(
            id="n-ontology-apply",
            type="ontology-apply",
            position={"x": 580, "y": 300},
            data={
                "label": "Apply Research Ontology",
                "config": {
                    "entity_types": [
                        "Paper",
                        "Author",
                        "Institution",
                        "Topic",
                        "Dataset",
                        "Method",
                    ],
                    "relationship_types": [
                        "AUTHORED",
                        "CITES",
                        "AFFILIATED_WITH",
                        "COVERS_TOPIC",
                        "USES_METHOD",
                        "USES_DATASET",
                    ],
                },
            },
        ),
        PipelineTemplateNode(
            id="n-entity-dedup",
            type="entity-dedup",
            position={"x": 840, "y": 300},
            data={
                "label": "Deduplicate Authors & Papers",
                "config": {
                    "strategy": "embedding",
                    "threshold": 0.90,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-similarity-build",
            type="similarity-build",
            position={"x": 1100, "y": 180},
            data={
                "label": "Build Similarity Index",
                "config": {
                    "model": "all-MiniLM-L6-v2",
                    "dimensions": 384,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-community-detect",
            type="community-detect",
            position={"x": 1100, "y": 420},
            data={
                "label": "Detect Research Clusters",
                "config": {
                    "algorithm": "leiden",
                    "resolution": 0.8,
                    "min_community_size": 2,
                    "max_levels": 4,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-community-summarize",
            type="community-summarize",
            position={"x": 1360, "y": 300},
            data={
                "label": "Summarise Research Areas",
                "config": {
                    "level": 0,
                    "max_tokens": 768,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-graph-output",
            type="graph-output",
            position={"x": 1620, "y": 300},
            data={
                "label": "Research Knowledge Base",
                "config": {
                    "graph_id": "from create-graph node",
                },
            },
        ),
    ],
    edges=[
        PipelineTemplateEdge(
            id="e-cg-iu", source="n-create-graph", target="n-ingest-url"
        ),
        PipelineTemplateEdge(
            id="e-cg-if", source="n-create-graph", target="n-ingest-file"
        ),
        PipelineTemplateEdge(
            id="e-iu-oa", source="n-ingest-url", target="n-ontology-apply"
        ),
        PipelineTemplateEdge(
            id="e-if-oa", source="n-ingest-file", target="n-ontology-apply"
        ),
        PipelineTemplateEdge(
            id="e-oa-ed", source="n-ontology-apply", target="n-entity-dedup"
        ),
        PipelineTemplateEdge(
            id="e-ed-sb", source="n-entity-dedup", target="n-similarity-build"
        ),
        PipelineTemplateEdge(
            id="e-ed-cd", source="n-entity-dedup", target="n-community-detect"
        ),
        PipelineTemplateEdge(
            id="e-sb-cs", source="n-similarity-build", target="n-community-summarize"
        ),
        PipelineTemplateEdge(
            id="e-cd-cs", source="n-community-detect", target="n-community-summarize"
        ),
        PipelineTemplateEdge(
            id="e-cs-go", source="n-community-summarize", target="n-graph-output"
        ),
    ],
)

_LEGAL_TEMPLATE = PipelineTemplate(
    id="legal-document-graph",
    name="Legal Document Graph",
    description=(
        "Parse contracts, regulations, and compliance documents into a structured legal graph. "
        "Extracts Regulation, Clause, Organization, Obligation, and Right entities; "
        "enforces a human-review gate before finalising; and groups clauses into thematic communities."
    ),
    domain="legal",
    tags=["legal", "compliance", "contracts", "regulations"],
    params=[
        PipelineTemplateParam(
            key="graph_name",
            label="Graph Name",
            description="Name for the legal knowledge graph",
            type="text",
            required=True,
        ),
        PipelineTemplateParam(
            key="llm_config_id",
            label="LLM Configuration",
            description="LLM config for clause and obligation extraction",
            type="select",
            required=True,
        ),
        PipelineTemplateParam(
            key="reviewer_email",
            label="Legal Reviewer E-mail",
            description="Who must approve the extracted graph before it is finalised",
            type="email",
            required=True,
        ),
    ],
    nodes=[
        PipelineTemplateNode(
            id="n-create-graph",
            type="create-graph",
            position={"x": 60, "y": 240},
            data={
                "label": "Create Graph",
                "config": {
                    "name": "{{graph_name}}",
                    "description": "Legal document and compliance knowledge graph",
                    "llm_config_id": "{{llm_config_id}}",
                    "ontology_template": "legal",
                },
            },
        ),
        PipelineTemplateNode(
            id="n-ingest-file",
            type="ingest-file",
            position={"x": 320, "y": 120},
            data={
                "label": "Ingest Legal Documents",
                "config": {
                    "mode": "full",
                },
            },
        ),
        PipelineTemplateNode(
            id="n-ontology-apply",
            type="ontology-apply",
            position={"x": 580, "y": 120},
            data={
                "label": "Apply Legal Ontology",
                "config": {
                    "entity_types": [
                        "Regulation",
                        "Clause",
                        "Organization",
                        "Obligation",
                        "Right",
                    ],
                    "relationship_types": [
                        "SUBJECT_TO",
                        "CONTAINS",
                        "IMPOSES",
                        "GRANTS",
                        "SUPERSEDES",
                    ],
                },
            },
        ),
        PipelineTemplateNode(
            id="n-entity-dedup",
            type="entity-dedup",
            position={"x": 840, "y": 120},
            data={
                "label": "Deduplicate Entities",
                "config": {
                    "strategy": "llm",
                    "threshold": 0.95,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-community-detect",
            type="community-detect",
            position={"x": 1100, "y": 120},
            data={
                "label": "Detect Legal Themes",
                "config": {
                    "algorithm": "leiden",
                    "resolution": 1.2,
                    "min_community_size": 2,
                    "max_levels": 2,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-review-gate",
            type="review-gate",
            position={"x": 1360, "y": 120},
            data={
                "label": "Legal Review Gate",
                "config": {
                    "instructions": (
                        "Please review the extracted legal entities and relationships. "
                        "Verify that obligations, rights, and parties are correctly identified "
                        "before the graph is published."
                    ),
                    "notify_email": "{{reviewer_email}}",
                },
            },
        ),
        PipelineTemplateNode(
            id="n-community-summarize",
            type="community-summarize",
            position={"x": 1620, "y": 120},
            data={
                "label": "Summarise Themes",
                "config": {
                    "level": 0,
                    "max_tokens": 512,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-graph-output",
            type="graph-output",
            position={"x": 1880, "y": 240},
            data={
                "label": "Legal Document Graph",
                "config": {
                    "graph_id": "from create-graph node",
                },
            },
        ),
    ],
    edges=[
        PipelineTemplateEdge(
            id="e-cg-if", source="n-create-graph", target="n-ingest-file"
        ),
        PipelineTemplateEdge(
            id="e-if-oa", source="n-ingest-file", target="n-ontology-apply"
        ),
        PipelineTemplateEdge(
            id="e-oa-ed", source="n-ontology-apply", target="n-entity-dedup"
        ),
        PipelineTemplateEdge(
            id="e-ed-cd", source="n-entity-dedup", target="n-community-detect"
        ),
        PipelineTemplateEdge(
            id="e-cd-rg", source="n-community-detect", target="n-review-gate"
        ),
        PipelineTemplateEdge(
            id="e-rg-cs",
            source="n-review-gate",
            target="n-community-summarize",
            type="trigger",
        ),
        PipelineTemplateEdge(
            id="e-cs-go", source="n-community-summarize", target="n-graph-output"
        ),
    ],
)

_CODEBASE_TEMPLATE = PipelineTemplate(
    id="codebase-architecture-graph",
    name="Codebase Architecture Graph",
    description=(
        "Ingest a code repository URL and build a structural knowledge graph of its architecture. "
        "Extracts Module, Class, Function, Import, and Dependency entities; "
        "builds a semantic similarity index for code search; "
        "and detects architectural clusters (e.g. domain layers, service boundaries)."
    ),
    domain="codebase",
    tags=["code", "architecture", "software", "dependencies"],
    params=[
        PipelineTemplateParam(
            key="graph_name",
            label="Graph Name",
            description="Name for the codebase knowledge graph",
            type="text",
            required=True,
        ),
        PipelineTemplateParam(
            key="repo_url",
            label="Repository URL",
            description="Public HTTPS URL of the git repository to ingest",
            type="url",
            required=True,
        ),
        PipelineTemplateParam(
            key="llm_config_id",
            label="LLM Configuration",
            description="LLM config for extracting code-level entities and relationships",
            type="select",
            required=True,
        ),
    ],
    nodes=[
        PipelineTemplateNode(
            id="n-create-graph",
            type="create-graph",
            position={"x": 60, "y": 240},
            data={
                "label": "Create Graph",
                "config": {
                    "name": "{{graph_name}}",
                    "description": "Codebase architecture and dependency graph",
                    "llm_config_id": "{{llm_config_id}}",
                    "ontology_template": "software",
                },
            },
        ),
        PipelineTemplateNode(
            id="n-ingest-url",
            type="ingest-url",
            position={"x": 320, "y": 120},
            data={
                "label": "Ingest Repository",
                "config": {
                    "url": "{{repo_url}}",
                    "mode": "full",
                },
            },
        ),
        PipelineTemplateNode(
            id="n-ontology-apply",
            type="ontology-apply",
            position={"x": 580, "y": 120},
            data={
                "label": "Apply Code Ontology",
                "config": {
                    "entity_types": [
                        "Module",
                        "Class",
                        "Function",
                        "Variable",
                        "Test",
                        "Dependency",
                    ],
                    "relationship_types": [
                        "IMPORTS",
                        "DEFINES",
                        "HAS_METHOD",
                        "CALLS",
                        "INHERITS",
                        "DEPENDS_ON",
                        "TESTS",
                    ],
                },
            },
        ),
        PipelineTemplateNode(
            id="n-entity-dedup",
            type="entity-dedup",
            position={"x": 840, "y": 120},
            data={
                "label": "Deduplicate Symbols",
                "config": {
                    "strategy": "embedding",
                    "threshold": 0.94,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-similarity-build",
            type="similarity-build",
            position={"x": 1100, "y": 60},
            data={
                "label": "Build Code Search Index",
                "config": {
                    "model": "all-MiniLM-L6-v2",
                    "dimensions": 384,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-community-detect",
            type="community-detect",
            position={"x": 1100, "y": 300},
            data={
                "label": "Detect Architectural Layers",
                "config": {
                    "algorithm": "leiden",
                    "resolution": 1.0,
                    "min_community_size": 3,
                    "max_levels": 3,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-community-summarize",
            type="community-summarize",
            position={"x": 1360, "y": 180},
            data={
                "label": "Summarise Layers",
                "config": {
                    "level": 0,
                    "max_tokens": 512,
                },
            },
        ),
        PipelineTemplateNode(
            id="n-graph-output",
            type="graph-output",
            position={"x": 1620, "y": 240},
            data={
                "label": "Codebase Architecture Graph",
                "config": {
                    "graph_id": "from create-graph node",
                },
            },
        ),
    ],
    edges=[
        PipelineTemplateEdge(
            id="e-cg-iu", source="n-create-graph", target="n-ingest-url"
        ),
        PipelineTemplateEdge(
            id="e-iu-oa", source="n-ingest-url", target="n-ontology-apply"
        ),
        PipelineTemplateEdge(
            id="e-oa-ed", source="n-ontology-apply", target="n-entity-dedup"
        ),
        PipelineTemplateEdge(
            id="e-ed-sb", source="n-entity-dedup", target="n-similarity-build"
        ),
        PipelineTemplateEdge(
            id="e-ed-cd", source="n-entity-dedup", target="n-community-detect"
        ),
        PipelineTemplateEdge(
            id="e-sb-cs", source="n-similarity-build", target="n-community-summarize"
        ),
        PipelineTemplateEdge(
            id="e-cd-cs", source="n-community-detect", target="n-community-summarize"
        ),
        PipelineTemplateEdge(
            id="e-cs-go", source="n-community-summarize", target="n-graph-output"
        ),
    ],
)

# ---------------------------------------------------------------------------
# Public catalogue dict — all built-in templates keyed by id
# ---------------------------------------------------------------------------

PIPELINE_TEMPLATES: dict[str, PipelineTemplate] = {
    t.id: t
    for t in [_HR_TEMPLATE, _RESEARCH_TEMPLATE, _LEGAL_TEMPLATE, _CODEBASE_TEMPLATE]
}
