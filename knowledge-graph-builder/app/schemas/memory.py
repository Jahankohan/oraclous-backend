"""
Memory API Schemas

Pydantic models for the Agent Memory API — store, retrieve, and forget
typed memories scoped to the Oraclous knowledge graph.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ==================== ENUMS ====================


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryScope(StrEnum):
    SESSION = "session"
    USER = "user"
    AGENT = "agent"
    TEAM = "team"
    ORGANIZATION = "organization"


class MemorySource(StrEnum):
    AGENT = "agent"
    USER_FEEDBACK = "user_feedback"
    INGESTION = "ingestion"
    INFERENCE = "inference"


class ContradictionResolution(StrEnum):
    NEW_WINS = "new_wins"
    OLD_WINS = "old_wins"
    UNRESOLVED = "unresolved"
    MERGED = "merged"


class MemoryRetrieverType(StrEnum):
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"
    GRAPH_TRAVERSAL = "graph_traversal"


class TemporalFilter(StrEnum):
    CURRENT = "current"
    ALL = "all"


# ==================== REQUEST SCHEMAS ====================


class MemoryCreate(BaseModel):
    """Request body for POST /graphs/{graphId}/memories"""

    type: MemoryType
    content: str = Field(..., min_length=1, max_length=10000)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    scope: MemoryScope = MemoryScope.AGENT
    agent_id: str | None = None
    session_id: str | None = None
    source: MemorySource = MemorySource.AGENT
    valid_from: datetime | None = None

    # Semantic-specific
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    is_negation: bool = False

    # Episodic-specific
    event_type: str | None = None
    user_id: str | None = None

    # Procedural-specific
    category: str | None = None
    trigger_pattern: str | None = None


class MemoryUpdate(BaseModel):
    """Request body for PATCH /graphs/{graphId}/memories/{memoryId}"""

    content: str | None = Field(default=None, min_length=1, max_length=10000)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None


# ==================== RESPONSE SCHEMAS ====================


class ConflictInfo(BaseModel):
    conflict_memory_id: str
    content: str
    resolution: ContradictionResolution


class MemoryCreateResponse(BaseModel):
    """Response for POST /graphs/{graphId}/memories"""

    memory_id: str
    importance_score: float
    contradictions_detected: list[ConflictInfo] = []
    entity_linked: str | None = None


class MemorySearchResult(BaseModel):
    """Single memory result in search response"""

    memory_id: str
    type: MemoryType
    content: str
    importance_score: float
    relevance_score: float
    confidence: float
    valid_from: datetime | None
    valid_to: datetime | None
    scope: MemoryScope
    agent_id: str | None = None
    session_id: str | None = None
    created_at: datetime | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0


class GraphFact(BaseModel):
    """Entity graph fact returned alongside memories"""

    subject: str
    predicate: str
    object: str
    source_chunk_id: str | None = None


class MemorySearchResponse(BaseModel):
    """Response for GET /graphs/{graphId}/memories/search"""

    memories: list[MemorySearchResult]
    graph_facts: list[GraphFact] = []
    total: int


class MemoryContext(BaseModel):
    """Response for GET /graphs/{graphId}/memories/context"""

    context_block: str
    memories_used: list[str]
    token_estimate: int
    retrieval_ms: int


class MemoryUpdateResponse(BaseModel):
    """Response for PATCH /graphs/{graphId}/memories/{memoryId}"""

    old_memory_id: str
    new_memory_id: str
    superseded_at: datetime


class ConsolidateResponse(BaseModel):
    """Response for POST /graphs/{graphId}/memories/consolidate"""

    job_id: str
    message: str
