"""
Pydantic schemas for the Assessment substrate (STORY-026, TASK-068).

Mirrors the Neo4j schema landed by TASK-067
(`app/cypher/migrations/2026-05-11_assessment_schema.cypher`). Field names
match Neo4j property names exactly — `finding_id`, not `findingId`; `run_id`,
not `runId`. This is required by the architecture rule "Pydantic schemas must
match Neo4j/Postgres model field names exactly" (see `oraclous-data-studio/CLAUDE.md`).

Three entity tiers:

1.  **Catalog-layer** (lives in `graph_id = '__assessments_catalog__'`):
    `AssessmentTemplate`, `Module`, `Source`. Templates and modules are shared
    across tenants; `:Source` is deduped so cross-run analyses ("all findings
    citing source X") work.

2.  **Run-layer** (lives in the customer's tenant `graph_id`):
    `AssessmentRun`, `ModuleRun`, `Subject`, `Finding`, `Conflict`,
    `Deliverable`, `UnresolvedQuestion`. ReBAC + Data Ownership clean.

3.  **Registry-layer** (per ADR-019; `curated`/`public` in `graph_id =
    '__registry__'`, `private` in the owner's tenant graph): `RegistryItem`.

The request/response wrappers at the bottom of this file are the shapes the
REST layer (TASK-069) and MCP wrappers (SPRINT-002) exchange with callers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# Reserved namespace constants — see ADR-018 + ADR-019.
# Importable by the service layer so callers do not hardcode the strings.
# =============================================================================

ASSESSMENTS_CATALOG_GRAPH_ID = "__assessments_catalog__"
REGISTRY_CATALOG_GRAPH_ID = "__registry__"


# =============================================================================
# Enum-like literals (mirror the values documented in STORY-026 §Graph Schema
# and the runtime states from the resumability acceptance criteria).
# =============================================================================

# Module classification per STORY-026 §Graph Schema:
#   kind ∈ {research, gap-research, analysis, synthesis, quality-gate}
ModuleKind = Literal[
    "research",
    "gap-research",
    "analysis",
    "synthesis",
    "quality-gate",
]

# Run / module-run lifecycle. STORY-026 §Acceptance Criteria specifies
# `planned → running → finished/failed`. `cancelled` is included because
# the orchestrator may abandon a planned wave when an upstream module fails.
RunStatus = Literal["planned", "running", "finished", "failed", "cancelled"]

# Findings are tagged with one of these labels per the existing Eurail
# `evidence.jsonl` schema (DIRECT = ground-truth claim, INFERRED = derived
# from a chain of reasoning, CONTRADICTION = flagged conflict source).
FindingLabel = Literal["DIRECT", "INFERRED", "CONTRADICTION"]

# Conflicts have a workflow status: open → resolved (in synthesis) or
# accepted_open (intentionally left unresolved for the deliverable).
ConflictStatus = Literal["open", "resolved", "accepted_open"]

# Open questions either get answered by a follow-up gap-research module
# (then `resolved`) or carry through to the unresolved-questions deliverable.
QuestionStatus = Literal["open", "resolved", "abandoned"]

# Deliverable kinds per STORY-026 §Graph Schema.
DeliverableKind = Literal["module-md", "final-md", "final-html", "final-pdf"]

# Registry tiers per ADR-019.
RegistryKind = Literal["skill", "agent", "tool", "mcp-server"]
RegistryVisibility = Literal["private", "public", "curated", "yanked"]


# =============================================================================
# Catalog-layer models — live in graph_id = '__assessments_catalog__'
# =============================================================================


class AssessmentTemplate(BaseModel):
    """A reusable assessment template (e.g. `assess-v1`, `eurail-report-v1`).

    Shared across tenants; stored in the catalog graph. Modules attach to a
    template via `(template)-[:HAS_MODULE]->(module)`.
    """

    model_config = ConfigDict(extra="forbid")

    template_id: str
    slug: str = Field(..., description="Stable human-readable identifier (e.g. 'eurail-report-v1').")
    name: str
    version: str
    vertical_slug: Optional[str] = Field(
        default=None,
        description="Vertical specialization, e.g. 'rail-cooperative'. None for generic assess-vN.",
    )
    description: Optional[str] = None
    created_at: Optional[datetime] = None


class Module(BaseModel):
    """A single research/analysis/synthesis/quality-gate stage within a template.

    Each module attaches to an `:Agent` row (existing platform agent) for the
    actual prompt/tools/model config, keeping module rows lightweight.
    """

    model_config = ConfigDict(extra="forbid")

    module_id: str
    template_id: str = Field(..., description="The template this module belongs to.")
    slug: str
    name: str
    wave: int = Field(..., ge=1, description="Wave number (1-based). All modules in a wave run in parallel.")
    ordinal: int = Field(..., ge=0, description="Display ordering inside the wave.")
    kind: ModuleKind
    agent_id: Optional[str] = Field(
        default=None,
        description="The `:Agent` row that carries the system prompt / model / tools for this module.",
    )
    description: Optional[str] = None


class Source(BaseModel):
    """A citation source. Deduped in the catalog graph by `source_id`/`url_normalized`.

    Per STORY-026 §Graph Schema and ADR-018 §Tenancy concession: sources live
    in the catalog graph so the admin `/assessments/findings:search?source_url=…`
    query can return findings across runs without breaking per-tenant ReBAC.
    The `[:CITES]` edge crosses namespace by `source_id` join at the API layer
    — never by Cypher MATCH across graphs.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    type: Optional[str] = Field(default=None, description="e.g. 'article', 'paper', 'press', 'webpage'.")
    url_normalized: Optional[str] = Field(
        default=None,
        description="Canonicalized URL used as the dedup key; None for non-web sources.",
    )
    name: Optional[str] = None
    publication_date: Optional[str] = None
    fetch_date: Optional[str] = None
    language: Optional[str] = None


# =============================================================================
# Run-layer models — live in the customer's tenant graph_id.
# Every model in this section carries `graph_id` because the service layer
# enforces it at write time and the indexes are composite on it.
# =============================================================================


class Subject(BaseModel):
    """The thing being assessed (a company, sector, regulation, etc.).

    Deduped per tenant by `slug` so Eurail = one row across all runs in that
    tenant.
    """

    model_config = ConfigDict(extra="forbid")

    subject_id: str
    graph_id: str
    slug: str
    name: str
    vertical_slug: Optional[str] = None
    domains: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class AssessmentRun(BaseModel):
    """A single execution of a template against a subject.

    `orchestrator_last_seen` is the heartbeat the Claude Code orchestrator
    pings at 60s intervals (STORY-026 §Acceptance Criteria — Resumability).
    A run whose `orchestrator_last_seen` is older than 5 minutes is
    considered orphaned and its `running` ModuleRuns get reset by the Celery
    reset agent.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    graph_id: str
    template_id: str
    subject_id: str
    status: RunStatus = "planned"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    orchestrator_last_seen: Optional[datetime] = None
    cli_flags: dict[str, Any] = Field(
        default_factory=dict,
        description="The CLI flags / config the orchestrator was invoked with.",
    )
    failure_reason: Optional[str] = None


class ModuleRun(BaseModel):
    """One (run × module) execution slot — the subagent's write target.

    Pre-created in `status='planned'` at run-creation time so the graph IS
    the backlog (STORY-026 §Coordination Model).
    """

    model_config = ConfigDict(extra="forbid")

    module_run_id: str
    graph_id: str
    run_id: str
    module_id: str
    wave: int = Field(..., ge=1)
    status: RunStatus = "planned"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    last_heartbeat_at: Optional[datetime] = Field(
        default=None,
        description="Subagent heartbeat; orphan detection threshold = 5 minutes (STORY-026).",
    )
    evidence_count: int = Field(default=0, ge=0)
    deliverable_path: Optional[str] = Field(
        default=None, description="Filesystem path for the module's MD output."
    )
    failure_reason: Optional[str] = None


class Finding(BaseModel):
    """A judged claim with provenance, label, confidence.

    Maps 1:1 to a row in the legacy `evidence.jsonl` from the filesystem-era
    assessment pipeline. The schema is deliberately permissive on `claim` and
    `raw` (LLM-generated text); the structured fields (`label`, `confidence`,
    `dimensions`) carry the queryable metadata.
    """

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    graph_id: str
    run_id: str
    module_run_id: str
    claim: str
    raw: Optional[str] = Field(default=None, description="Verbatim quote or raw extract text.")
    label: FindingLabel = "DIRECT"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dimensions: list[str] = Field(
        default_factory=list,
        description="Free-form dimension tags (e.g. 'regulatory', 'tech-maturity').",
    )
    ai_adoption_relevance: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Optional secondary score specific to AI-adoption assessments.",
    )
    notes: Optional[str] = None
    superseded_by: Optional[str] = Field(
        default=None, description="finding_id of a later, more authoritative finding."
    )
    # Citation — denormalized for write-side simplicity. The service layer
    # MERGEs the `:Source` in the catalog graph and creates the `[:CITES]` edge.
    source_id: Optional[str] = None
    source_quote: Optional[str] = Field(
        default=None, description="The exact passage cited (lands on the [:CITES] edge)."
    )
    source_locator: Optional[str] = Field(
        default=None, description="Page/timestamp/anchor (lands on the [:CITES] edge)."
    )


class Conflict(BaseModel):
    """A detected disagreement among findings within a run."""

    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    graph_id: str
    run_id: str
    topic: str
    summary: str
    status: ConflictStatus = "open"
    resolution: Optional[str] = None
    synthesis_note: Optional[str] = None
    involved_finding_ids: list[str] = Field(
        default_factory=list,
        description="finding_ids that participate in this conflict. The service creates one [:INVOLVES] edge per id.",
    )


class Deliverable(BaseModel):
    """An artifact produced by a module or by the final synthesis stage.

    `content_uri` is a path/URI string for SPRINT-001 (filesystem placeholder).
    SPRINT-002 will introduce a Postgres-backed `:Blob` CAS keyed by `sha256`.
    Small markdown payloads may also be stored inline via `content_inline`
    for early access patterns; readers should fall back to fetching by
    `content_uri` if `content_inline` is empty.
    """

    model_config = ConfigDict(extra="forbid")

    deliverable_id: str
    graph_id: str
    run_id: str
    module_run_id: Optional[str] = Field(
        default=None,
        description="The module that produced this artifact, when applicable. None for final-* kinds.",
    )
    kind: DeliverableKind
    filename: str
    ordinal: int = Field(default=0, ge=0)
    content_uri: Optional[str] = Field(
        default=None,
        description="Path/URI to the rendered artifact (SPRINT-001 filesystem placeholder).",
    )
    content_inline: Optional[str] = Field(
        default=None,
        description=(
            "Optional inline content for small markdown payloads "
            "(< ~50KB). SPRINT-002 introduces :Blob CAS for large payloads."
        ),
    )
    sha256: Optional[str] = None
    word_count: Optional[int] = Field(default=None, ge=0)


class UnresolvedQuestion(BaseModel):
    """A question raised by a research subagent that needs gap-research follow-up."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    graph_id: str
    run_id: str
    module_run_id: str
    text: str
    suggested_module: Optional[str] = Field(
        default=None,
        description="Slug of the module that should answer this question, if known.",
    )
    status: QuestionStatus = "open"


# =============================================================================
# Registry-layer model — ADR-019
# =============================================================================


class RegistryItem(BaseModel):
    """A Registry entry: skill, agent definition, tool, or MCP server.

    Per ADR-019, `private` items live in the owner's tenant graph; `curated`
    and `public` items live in `__registry__`. The service layer routes the
    write based on `visibility`.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    graph_id: str = Field(
        ...,
        description=(
            "Tenant graph_id for `private` items; '__registry__' for `curated` and `public`. "
            "Validated by `_validate_registry_placement()` at the service layer."
        ),
    )
    kind: RegistryKind
    slug: str = Field(
        ...,
        description=(
            "Catalog namespace: `users/<owner_user_id>/<slug>` for private, flat for curated+public. "
            "Slug uniqueness is per (kind, visibility) tier per ADR-019."
        ),
    )
    version: str = Field(default="0.1.0")
    visibility: RegistryVisibility = "private"
    owner_user_id: str
    name: str
    description: Optional[str] = None
    content_uri: Optional[str] = None
    sha256: Optional[str] = None
    created_at: Optional[datetime] = None
    yanked_at: Optional[datetime] = None

    @field_validator("yanked_at")
    @classmethod
    def _yanked_at_requires_visibility(
        cls, v: Optional[datetime], info: Any
    ) -> Optional[datetime]:
        """ADR-019: yanked is reachable from public; ensure consistency."""
        if v is not None and info.data.get("visibility") not in ("yanked", "public"):
            raise ValueError(
                "yanked_at can only be set when visibility is 'yanked' or 'public' (during the transition)"
            )
        return v


# =============================================================================
# Request / response wrappers — the REST layer (TASK-069) speaks these shapes
# =============================================================================


class CreateRunRequest(BaseModel):
    """Input to `AssessmentService.create_run()`.

    `template_slug` resolves against the catalog graph; the service looks up
    the `template_id` and the attached `:Module` rows. `subject` may carry
    just `slug + name` if the Subject doesn't yet exist in the tenant graph
    (the service MERGEs it).
    """

    model_config = ConfigDict(extra="forbid")

    template_slug: str
    subject: Subject
    cli_flags: dict[str, Any] = Field(default_factory=dict)
    run_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional client-supplied UUID for idempotent retry. If a run with this id "
            "already exists in the tenant graph, `create_run()` returns the existing run."
        ),
    )


class CreateRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    template_id: str
    subject_id: str
    module_run_ids: list[str]
    status: RunStatus
    already_existed: bool = Field(
        default=False,
        description="True when the request matched an existing run (idempotent replay).",
    )


class UpdateModuleRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Optional[RunStatus] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    last_heartbeat_at: Optional[datetime] = None
    evidence_count: Optional[int] = Field(default=None, ge=0)
    deliverable_path: Optional[str] = None
    failure_reason: Optional[str] = None


class RecordFindingBulkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[Finding]


class BulkItemResult(BaseModel):
    """Per-item result for bulk writes.

    The bulk-response shape decision (STORY-026 Open Question #4): we chose
    per-record success/failure over all-or-nothing. Subagents writing 50+
    findings in a batch should not have one bad row roll back the whole
    request. Idempotency replay is also cleaner — already-merged rows return
    `success=True, already_existed=True` instead of failing.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="The natural id of the row (finding_id, conflict_id, …).")
    success: bool
    already_existed: bool = False
    error: Optional[str] = None


class BulkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    succeeded: int
    failed: int
    results: list[BulkItemResult]


class RecordConflictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conflict: Conflict


class RecordUnresolvedQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: UnresolvedQuestion


class PersistDeliverableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deliverable: Deliverable


class PersistFinalDocsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deliverables: list[Deliverable]


class FinalizeRunResponse(BaseModel):
    """Output of `finalize_run()` — server-side citation-coverage gate.

    The `passed` flag reflects whether the run met the minimum thresholds
    (direct-finding count, deliverable count, citation-coverage ratio). The
    service flips `:AssessmentRun.status` to `finished` when `passed=True`,
    `failed` otherwise. `finished_at` is the timestamp the service applied.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    passed: bool
    status: RunStatus
    finished_at: datetime
    direct_finding_count: int
    inferred_finding_count: int
    deliverable_count: int
    unresolved_conflict_count: int
    open_question_count: int
    failure_reasons: list[str] = Field(default_factory=list)
