"""Pydantic schemas for cross-graph federation endpoints."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

MAX_GRAPH_IDS = 10
MAX_RESULTS_PER_GRAPH = 100
MAX_TOTAL_RESULTS = 500
QUERY_TIMEOUT_MS = 8000


# ─── Request models ───────────────────────────────────────────────────────────

class FederatedQueryOptions(BaseModel):
    deduplicate_entities: bool = True
    max_results_per_graph: int = Field(default=50, ge=1, le=MAX_RESULTS_PER_GRAPH)
    include_cross_graph_links: bool = True
    timeout_ms: int = Field(default=QUERY_TIMEOUT_MS, ge=1000, le=QUERY_TIMEOUT_MS)


class FederatedQueryRequest(BaseModel):
    graph_ids: List[str] = Field(..., min_length=2, max_length=MAX_GRAPH_IDS)
    query: str = Field(..., min_length=1, max_length=2000)
    options: FederatedQueryOptions = Field(default_factory=FederatedQueryOptions)

    @field_validator("graph_ids")
    @classmethod
    def no_duplicate_graph_ids(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("graph_ids must be unique")
        return v


class FederatedVectorSearchRequest(BaseModel):
    graph_ids: List[str] = Field(..., min_length=2, max_length=MAX_GRAPH_IDS)
    query_text: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=20, ge=1, le=MAX_RESULTS_PER_GRAPH)
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    @field_validator("graph_ids")
    @classmethod
    def no_duplicate_graph_ids(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("graph_ids must be unique")
        return v


# ─── Response models ──────────────────────────────────────────────────────────

class FederatedEntity(BaseModel):
    entity_id: str
    name: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
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
    graphs_skipped: List[str] = Field(default_factory=list)
    timed_out: bool = False
    deduplication_status: str = "not_requested"  # "not_requested" | "pending" | "complete"


class FederatedQueryResponse(BaseModel):
    status: str
    graphs_queried: List[str]
    total_entities: int
    entities: List[FederatedEntity]
    cross_graph_links: List[CrossGraphLink] = Field(default_factory=list)
    query_meta: QueryMeta


class FederatedVectorResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_graph_id: str
    source_graph_name: str
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None


class FederatedVectorSearchResponse(BaseModel):
    status: str
    graphs_queried: List[str]
    total_results: int
    results: List[FederatedVectorResult]
    query_meta: QueryMeta
