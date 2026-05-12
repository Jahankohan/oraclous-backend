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

Size policy (TASK-076)
----------------------

Every string field declares an explicit ``max_length`` and every list field
declares an explicit ``max_length`` (Pydantic v2 unifies the constraint name
across ``str`` and ``list``; what older Pydantic called ``max_items`` is just
``max_length`` here). The defense is against property-bloat / DoS via an
adversarial caller submitting unbounded payloads that we then write into
Neo4j as node properties — closing the MEDIUM finding (#3) from TASK-073.

The bounds use module-level ``SIZE_*`` and ``LIST_MAX_*`` constants so every
new field inherits the policy by picking the right tier rather than inventing
a new number. If you find yourself reaching for an inline literal, add a tier
or extend an existing one instead.

| Tier               | Bound  | When to use                                         |
|--------------------|--------|-----------------------------------------------------|
| ``SIZE_ID``        |    128 | Natural ids (``run_id``, ``finding_id``, etc.)      |
| ``SIZE_SLUG``      |    128 | Stable kebab-case slugs                             |
| ``SIZE_ENUM``      |     64 | Short enum-like strings outside a ``Literal``       |
| ``SIZE_LANG``      |     16 | ISO language / locale codes                         |
| ``SIZE_DATE``      |     32 | ISO date / datetime strings                         |
| ``SIZE_VERSION``   |     32 | SemVer-shaped strings                               |
| ``SIZE_HASH``      |    128 | Hex digests (sha256 etc.)                           |
| ``SIZE_NAME``      |    256 | Human-facing names / titles                         |
| ``SIZE_URL``       |  2_048 | URLs / URIs (canonical and raw)                     |
| ``SIZE_FILENAME``  |  1_024 | File / path basenames                               |
| ``SIZE_SHORT_TEXT``|  4_096 | Short prose: ``description``, ``topic``, errors     |
| ``SIZE_CLAIM``     |  8_192 | Single-sentence claims, ``source_quote``            |
| ``SIZE_LONG_TEXT`` | 16_384 | Long prose: ``notes``, ``summary``, ``resolution``  |
| ``SIZE_BLOB_TEXT`` | 65_536 | ``raw`` extracts, ``content_inline`` deliverables   |
|                    |        | (~50 KB; matches the inline / CAS cutoff)           |
| ``LIST_MAX_TAGS``  |     32 | Small free-form tag lists (``dimensions``)          |
| ``LIST_MAX_IDS``   |    256 | Id-ref lists (``involved_finding_ids``)             |
| ``LIST_MAX_ITEMS`` |    512 | Bulk-write payloads (``findings``, ``deliverables``)|

Real-world reality check: the largest values observed in the Eurail
2026-05-06 backfill run are ``claim`` 509, ``raw`` 775, ``notes`` 275,
``resolution`` 358, ``synthesis_note`` 607, ``url`` 173, and the largest
module-md deliverable inline payload is 43_876 chars — every bound above has
substantial headroom over real data. ``content_inline`` is sized at 65_536
specifically so the Eurail deliverable round-trips lossless.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# Reserved namespace constants — see ADR-018 + ADR-019.
# Importable by the service layer so callers do not hardcode the strings.
# =============================================================================

ASSESSMENTS_CATALOG_GRAPH_ID = "__assessments_catalog__"
REGISTRY_CATALOG_GRAPH_ID = "__registry__"


# =============================================================================
# Size-policy constants (TASK-076). See the module docstring for the table.
# Importable by tests and future schema authors; never inline a literal.
# =============================================================================

SIZE_ID = 128
SIZE_SLUG = 128
SIZE_ENUM = 64
SIZE_LANG = 16
SIZE_DATE = 32
SIZE_VERSION = 32
SIZE_HASH = 128
SIZE_NAME = 256
SIZE_URL = 2_048
SIZE_FILENAME = 1_024
SIZE_SHORT_TEXT = 4_096
SIZE_CLAIM = 8_192
SIZE_LONG_TEXT = 16_384
SIZE_BLOB_TEXT = 65_536

LIST_MAX_TAGS = 32
LIST_MAX_IDS = 256
LIST_MAX_ITEMS = 512


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

    template_id: str = Field(..., max_length=SIZE_ID)
    slug: str = Field(
        ...,
        max_length=SIZE_SLUG,
        description="Stable human-readable identifier (e.g. 'eurail-report-v1').",
    )
    name: str = Field(..., max_length=SIZE_NAME)
    version: str = Field(..., max_length=SIZE_VERSION)
    vertical_slug: str | None = Field(
        default=None,
        max_length=SIZE_SLUG,
        description="Vertical specialization, e.g. 'rail-cooperative'. None for generic assess-vN.",
    )
    description: str | None = Field(default=None, max_length=SIZE_SHORT_TEXT)
    created_at: datetime | None = None


class Module(BaseModel):
    """A single research/analysis/synthesis/quality-gate stage within a template.

    Each module attaches to an `:Agent` row (existing platform agent) for the
    actual prompt/tools/model config, keeping module rows lightweight.
    """

    model_config = ConfigDict(extra="forbid")

    module_id: str = Field(..., max_length=SIZE_ID)
    template_id: str = Field(
        ...,
        max_length=SIZE_ID,
        description="The template this module belongs to.",
    )
    slug: str = Field(..., max_length=SIZE_SLUG)
    name: str = Field(..., max_length=SIZE_NAME)
    wave: int = Field(
        ...,
        ge=1,
        description="Wave number (1-based). All modules in a wave run in parallel.",
    )
    ordinal: int = Field(..., ge=0, description="Display ordering inside the wave.")
    kind: ModuleKind
    agent_id: str | None = Field(
        default=None,
        max_length=SIZE_ID,
        description="The `:Agent` row that carries the system prompt / model / tools for this module.",
    )
    description: str | None = Field(default=None, max_length=SIZE_SHORT_TEXT)


class Source(BaseModel):
    """A citation source. Deduped in the catalog graph by `source_id`/`url_normalized`.

    Per STORY-026 §Graph Schema and ADR-018 §Tenancy concession: sources live
    in the catalog graph so the admin `/assessments/findings:search?source_url=…`
    query can return findings across runs without breaking per-tenant ReBAC.
    The `[:CITES]` edge crosses namespace by `source_id` join at the API layer
    — never by Cypher MATCH across graphs.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., max_length=SIZE_ID)
    type: str | None = Field(
        default=None,
        max_length=SIZE_ENUM,
        description="e.g. 'article', 'paper', 'press', 'webpage'.",
    )
    url_normalized: str | None = Field(
        default=None,
        max_length=SIZE_URL,
        description="Canonicalized URL used as the dedup key; None for non-web sources.",
    )
    name: str | None = Field(default=None, max_length=SIZE_NAME)
    publication_date: str | None = Field(default=None, max_length=SIZE_DATE)
    fetch_date: str | None = Field(default=None, max_length=SIZE_DATE)
    language: str | None = Field(default=None, max_length=SIZE_LANG)


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

    subject_id: str = Field(..., max_length=SIZE_ID)
    graph_id: str = Field(..., max_length=SIZE_ID)
    slug: str = Field(..., max_length=SIZE_SLUG)
    name: str = Field(..., max_length=SIZE_NAME)
    vertical_slug: str | None = Field(default=None, max_length=SIZE_SLUG)
    domains: list[str] = Field(default_factory=list, max_length=LIST_MAX_TAGS)
    aliases: list[str] = Field(default_factory=list, max_length=LIST_MAX_TAGS)


class AssessmentRun(BaseModel):
    """A single execution of a template against a subject.

    `orchestrator_last_seen` is the heartbeat the Claude Code orchestrator
    pings at 60s intervals (STORY-026 §Acceptance Criteria — Resumability).
    A run whose `orchestrator_last_seen` is older than 5 minutes is
    considered orphaned and its `running` ModuleRuns get reset by the Celery
    reset agent.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., max_length=SIZE_ID)
    graph_id: str = Field(..., max_length=SIZE_ID)
    template_id: str = Field(..., max_length=SIZE_ID)
    subject_id: str = Field(..., max_length=SIZE_ID)
    status: RunStatus = "planned"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    orchestrator_last_seen: datetime | None = None
    cli_flags: dict[str, Any] = Field(
        default_factory=dict,
        description="The CLI flags / config the orchestrator was invoked with.",
    )
    failure_reason: str | None = Field(default=None, max_length=SIZE_SHORT_TEXT)


class ModuleRun(BaseModel):
    """One (run × module) execution slot — the subagent's write target.

    Pre-created in `status='planned'` at run-creation time so the graph IS
    the backlog (STORY-026 §Coordination Model).
    """

    model_config = ConfigDict(extra="forbid")

    module_run_id: str = Field(..., max_length=SIZE_ID)
    graph_id: str = Field(..., max_length=SIZE_ID)
    run_id: str = Field(..., max_length=SIZE_ID)
    module_id: str = Field(..., max_length=SIZE_ID)
    wave: int = Field(..., ge=1)
    status: RunStatus = "planned"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_heartbeat_at: datetime | None = Field(
        default=None,
        description="Subagent heartbeat; orphan detection threshold = 5 minutes (STORY-026).",
    )
    evidence_count: int = Field(default=0, ge=0)
    deliverable_path: str | None = Field(
        default=None,
        max_length=SIZE_URL,
        description="Filesystem path for the module's MD output.",
    )
    failure_reason: str | None = Field(default=None, max_length=SIZE_SHORT_TEXT)


class Finding(BaseModel):
    """A judged claim with provenance, label, confidence.

    Maps 1:1 to a row in the legacy `evidence.jsonl` from the filesystem-era
    assessment pipeline. The schema is deliberately permissive on `claim` and
    `raw` (LLM-generated text); the structured fields (`label`, `confidence`,
    `dimensions`) carry the queryable metadata.
    """

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(..., max_length=SIZE_ID)
    graph_id: str = Field(..., max_length=SIZE_ID)
    run_id: str = Field(..., max_length=SIZE_ID)
    module_run_id: str = Field(..., max_length=SIZE_ID)
    claim: str = Field(..., max_length=SIZE_CLAIM)
    raw: str | None = Field(
        default=None,
        max_length=SIZE_BLOB_TEXT,
        description="Verbatim quote or raw extract text.",
    )
    label: FindingLabel = "DIRECT"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dimensions: list[str] = Field(
        default_factory=list,
        max_length=LIST_MAX_TAGS,
        description="Free-form dimension tags (e.g. 'regulatory', 'tech-maturity').",
    )
    ai_adoption_relevance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional secondary score specific to AI-adoption assessments.",
    )
    notes: str | None = Field(default=None, max_length=SIZE_LONG_TEXT)
    superseded_by: str | None = Field(
        default=None,
        max_length=SIZE_ID,
        description="finding_id of a later, more authoritative finding.",
    )
    # Citation — denormalized for write-side simplicity. The service layer
    # MERGEs the `:Source` in the catalog graph and creates the `[:CITES]` edge.
    source_id: str | None = Field(default=None, max_length=SIZE_ID)
    source_quote: str | None = Field(
        default=None,
        max_length=SIZE_CLAIM,
        description="The exact passage cited (lands on the [:CITES] edge).",
    )
    source_locator: str | None = Field(
        default=None,
        max_length=SIZE_SHORT_TEXT,
        description="Page/timestamp/anchor (lands on the [:CITES] edge).",
    )
    # Optional nested Source payload (TASK-077). When populated, the service
    # threads the fields into the catalog `:Source` MERGE so cross-run search
    # by URL/name/date works. `source.source_id` MUST equal `source_id` above
    # when both are set; the service enforces this and rejects the write on
    # mismatch (surfaced as per-record failure in the bulk response). Old
    # callers that only supply `source_id` continue to work — the catalog row
    # just won't carry URL/name/date for those.
    source: Source | None = Field(
        default=None,
        description=(
            "Full Source metadata to write into the catalog graph on first "
            "observation. Optional; required only when the caller wants the "
            "catalog `:Source` row populated with URL / name / dates."
        ),
    )


class Conflict(BaseModel):
    """A detected disagreement among findings within a run."""

    model_config = ConfigDict(extra="forbid")

    conflict_id: str = Field(..., max_length=SIZE_ID)
    graph_id: str = Field(..., max_length=SIZE_ID)
    run_id: str = Field(..., max_length=SIZE_ID)
    topic: str = Field(..., max_length=SIZE_SHORT_TEXT)
    summary: str = Field(..., max_length=SIZE_LONG_TEXT)
    status: ConflictStatus = "open"
    resolution: str | None = Field(default=None, max_length=SIZE_LONG_TEXT)
    synthesis_note: str | None = Field(default=None, max_length=SIZE_LONG_TEXT)
    involved_finding_ids: list[str] = Field(
        default_factory=list,
        max_length=LIST_MAX_IDS,
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

    deliverable_id: str = Field(..., max_length=SIZE_ID)
    graph_id: str = Field(..., max_length=SIZE_ID)
    run_id: str = Field(..., max_length=SIZE_ID)
    module_run_id: str | None = Field(
        default=None,
        max_length=SIZE_ID,
        description="The module that produced this artifact, when applicable. None for final-* kinds.",
    )
    kind: DeliverableKind
    filename: str = Field(..., max_length=SIZE_FILENAME)
    ordinal: int = Field(default=0, ge=0)
    content_uri: str | None = Field(
        default=None,
        max_length=SIZE_URL,
        description="Path/URI to the rendered artifact (SPRINT-001 filesystem placeholder).",
    )
    content_inline: str | None = Field(
        default=None,
        max_length=SIZE_BLOB_TEXT,
        description=(
            "Optional inline content for small markdown payloads "
            "(~50 KB ceiling — SIZE_BLOB_TEXT). SPRINT-002 introduces "
            ":Blob CAS for larger payloads."
        ),
    )
    sha256: str | None = Field(default=None, max_length=SIZE_HASH)
    word_count: int | None = Field(default=None, ge=0)


class UnresolvedQuestion(BaseModel):
    """A question raised by a research subagent that needs gap-research follow-up."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(..., max_length=SIZE_ID)
    graph_id: str = Field(..., max_length=SIZE_ID)
    run_id: str = Field(..., max_length=SIZE_ID)
    module_run_id: str = Field(..., max_length=SIZE_ID)
    text: str = Field(..., max_length=SIZE_LONG_TEXT)
    suggested_module: str | None = Field(
        default=None,
        max_length=SIZE_SLUG,
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

    item_id: str = Field(..., max_length=SIZE_ID)
    graph_id: str = Field(
        ...,
        max_length=SIZE_ID,
        description=(
            "Tenant graph_id for `private` items; '__registry__' for `curated` and `public`. "
            "Validated by `_validate_registry_placement()` at the service layer."
        ),
    )
    kind: RegistryKind
    slug: str = Field(
        ...,
        max_length=SIZE_SLUG,
        description=(
            "Catalog namespace: `users/<owner_user_id>/<slug>` for private, flat for curated+public. "
            "Slug uniqueness is per (kind, visibility) tier per ADR-019."
        ),
    )
    version: str = Field(default="0.1.0", max_length=SIZE_VERSION)
    visibility: RegistryVisibility = "private"
    owner_user_id: str = Field(..., max_length=SIZE_ID)
    name: str = Field(..., max_length=SIZE_NAME)
    description: str | None = Field(default=None, max_length=SIZE_SHORT_TEXT)
    content_uri: str | None = Field(default=None, max_length=SIZE_URL)
    sha256: str | None = Field(default=None, max_length=SIZE_HASH)
    created_at: datetime | None = None
    yanked_at: datetime | None = None

    @field_validator("yanked_at")
    @classmethod
    def _yanked_at_requires_visibility(
        cls, v: datetime | None, info: Any
    ) -> datetime | None:
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

    template_slug: str = Field(..., max_length=SIZE_SLUG)
    subject: Subject
    cli_flags: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = Field(
        default=None,
        max_length=SIZE_ID,
        description=(
            "Optional client-supplied UUID for idempotent retry. If a run with this id "
            "already exists in the tenant graph, `create_run()` returns the existing run."
        ),
    )


class CreateRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., max_length=SIZE_ID)
    template_id: str = Field(..., max_length=SIZE_ID)
    subject_id: str = Field(..., max_length=SIZE_ID)
    module_run_ids: list[str] = Field(default_factory=list, max_length=LIST_MAX_IDS)
    status: RunStatus
    already_existed: bool = Field(
        default=False,
        description="True when the request matched an existing run (idempotent replay).",
    )


class UpdateModuleRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RunStatus | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    evidence_count: int | None = Field(default=None, ge=0)
    deliverable_path: str | None = Field(default=None, max_length=SIZE_URL)
    failure_reason: str | None = Field(default=None, max_length=SIZE_SHORT_TEXT)


class RecordFindingBulkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(..., max_length=LIST_MAX_ITEMS)


class BulkItemResult(BaseModel):
    """Per-item result for bulk writes.

    The bulk-response shape decision (STORY-026 Open Question #4): we chose
    per-record success/failure over all-or-nothing. Subagents writing 50+
    findings in a batch should not have one bad row roll back the whole
    request. Idempotency replay is also cleaner — already-merged rows return
    `success=True, already_existed=True` instead of failing.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        max_length=SIZE_ID,
        description="The natural id of the row (finding_id, conflict_id, …).",
    )
    success: bool
    already_existed: bool = False
    error: str | None = Field(default=None, max_length=SIZE_SHORT_TEXT)


class BulkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    succeeded: int
    failed: int
    results: list[BulkItemResult] = Field(
        default_factory=list, max_length=LIST_MAX_ITEMS
    )


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

    deliverables: list[Deliverable] = Field(..., max_length=LIST_MAX_ITEMS)


class FinalizeRunResponse(BaseModel):
    """Output of `finalize_run()` — server-side citation-coverage gate.

    The `passed` flag reflects whether the run met the minimum thresholds
    (direct-finding count, deliverable count, citation-coverage ratio). The
    service flips `:AssessmentRun.status` to `finished` when `passed=True`,
    `failed` otherwise. `finished_at` is the timestamp the service applied.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., max_length=SIZE_ID)
    passed: bool
    status: RunStatus
    finished_at: datetime
    direct_finding_count: int
    inferred_finding_count: int
    deliverable_count: int
    unresolved_conflict_count: int
    open_question_count: int
    failure_reasons: list[str] = Field(default_factory=list, max_length=LIST_MAX_TAGS)
