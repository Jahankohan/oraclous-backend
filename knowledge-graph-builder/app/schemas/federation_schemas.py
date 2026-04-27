"""Pydantic schemas for cross-graph federation endpoints."""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

MAX_GRAPH_IDS = 10
MAX_RESULTS_PER_GRAPH = 100
MAX_TOTAL_RESULTS = 500
QUERY_TIMEOUT_MS = 8000


# ─── SAME_AS candidate type ───────────────────────────────────────────────────


class SameAsCandidate(TypedDict):
    """A candidate entity pair for SAME_AS resolution.

    Produced by find_same_as_candidates; consumed by TASK-010's scoring step.
    SAME_AS links are NOT created here — only candidates are returned.
    """

    entity: dict[str, Any]
    score: float
    method: Literal["exact", "vector"]


# ─── Request models ───────────────────────────────────────────────────────────


class FederatedQueryOptions(BaseModel):
    deduplicate_entities: bool = True
    max_results_per_graph: int = Field(default=50, ge=1, le=MAX_RESULTS_PER_GRAPH)
    include_cross_graph_links: bool = True
    timeout_ms: int = Field(default=QUERY_TIMEOUT_MS, ge=1000, le=QUERY_TIMEOUT_MS)


class FederatedQueryRequest(BaseModel):
    graph_ids: list[str] = Field(..., min_length=2, max_length=MAX_GRAPH_IDS)
    query: str = Field(..., min_length=1, max_length=2000)
    options: FederatedQueryOptions = Field(default_factory=FederatedQueryOptions)

    @field_validator("graph_ids")
    @classmethod
    def no_duplicate_graph_ids(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("graph_ids must be unique")
        return v


class FederatedVectorSearchRequest(BaseModel):
    graph_ids: list[str] = Field(..., min_length=2, max_length=MAX_GRAPH_IDS)
    query_text: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=20, ge=1, le=MAX_RESULTS_PER_GRAPH)
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    @field_validator("graph_ids")
    @classmethod
    def no_duplicate_graph_ids(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("graph_ids must be unique")
        return v


# ─── Response models ──────────────────────────────────────────────────────────


class FederatedEntity(BaseModel):
    entity_id: str
    name: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source_graph_id: str
    source_graph_name: str


class CrossGraphLink(BaseModel):
    entity_a_id: str
    entity_b_id: str
    link_type: str  # "SAME_AS"
    confidence: float
    graph_a: str
    graph_b: str


class QueryMeta(BaseModel):
    execution_time_ms: int
    graphs_skipped: list[str] = Field(default_factory=list)
    timed_out: bool = False
    deduplication_status: str = (
        "not_requested"  # "not_requested" | "pending" | "complete"
    )


class FederatedQueryResponse(BaseModel):
    status: str
    graphs_queried: list[str]
    total_entities: int
    entities: list[FederatedEntity]
    cross_graph_links: list[CrossGraphLink] = Field(default_factory=list)
    query_meta: QueryMeta


class FederatedVectorResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_graph_id: str
    source_graph_name: str
    entity_name: str | None = None
    entity_type: str | None = None


class FederatedVectorSearchResponse(BaseModel):
    status: str
    graphs_queried: list[str]
    total_results: int
    results: list[FederatedVectorResult]
    query_meta: QueryMeta


# ─── Federation candidates endpoint schemas (TASK-016) ───────────────────────


class FederationCandidatesRequest(BaseModel):
    """Request body for POST /graphs/{graph_id}/federation/candidates."""

    target_graph_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_GRAPH_IDS,
        description="Graph IDs to search for SAME_AS candidates against graph_id.",
        examples=[["graph-Y", "graph-Z"]],
    )
    entity_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="Optional name filter. When provided, only entities whose name "
        "contains this string (case-insensitive) are considered.",
        examples=["Acme Corp"],
    )

    @field_validator("target_graph_ids")
    @classmethod
    def no_duplicate_target_graph_ids(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("target_graph_ids must be unique")
        return v


class SignalScores(BaseModel):
    """Per-signal breakdown of a federation candidate score."""

    embedding: float = Field(
        ...,
        description="Cosine similarity of entity embeddings (0.0 = not available).",
    )
    name: float = Field(..., description="Normalised name-match score.")
    type: float = Field(
        ...,
        description="Type-match score (1.0 = exact match, 0.0 = mismatch or missing).",
    )
    shared_relations: float = Field(
        ...,
        description="Fraction of shared relation types (0.0 = not yet computed).",
    )


class FederationCandidateResult(BaseModel):
    """A single SAME_AS candidate pair returned by the candidates endpoint."""

    entity_a: dict[str, Any] = Field(
        ...,
        description="Source entity: {id, name, graph_id}.",
        examples=[{"id": "4:abc:1", "name": "Acme Corp", "graph_id": "graph-X"}],
    )
    entity_b: dict[str, Any] = Field(
        ...,
        description="Target entity: {id, name, graph_id}.",
        examples=[{"id": "4:def:2", "name": "Acme Corp", "graph_id": "graph-Y"}],
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Combined SAME_AS confidence score. Only candidates with score >= 0.60 "
            "are returned by this endpoint."
        ),
    )
    signals: SignalScores


# ─── Federation resolve schemas (TASK-017) ────────────────────────────────────


class FederationResolveRequest(BaseModel):
    """Request body for POST /graphs/{graph_id}/federation/resolve."""

    target_graph_id: str = Field(
        ...,
        description="ID of the target graph to resolve entities against",
        examples=["graph-Y"],
    )
    confidence_threshold: float = Field(
        default=0.85,
        description="Minimum confidence score for creating SAME_AS links. Clamped to [0.60, 1.0].",
        examples=[0.85],
    )

    @field_validator("confidence_threshold")
    @classmethod
    def clamp_threshold(cls, v: float) -> float:
        return max(0.60, min(1.0, v))


class FederationResolveResponse(BaseModel):
    """Immediate response for POST /graphs/{graph_id}/federation/resolve."""

    task_id: str = Field(
        ...,
        description="Celery task ID; poll GET /tasks/{task_id} for result",
        examples=["3c8a2b4e-1234-5678-abcd-ef0123456789"],
    )
    status: str = Field(
        default="queued",
        description="Task dispatch status; always 'queued' on success",
        examples=["queued"],
    )
