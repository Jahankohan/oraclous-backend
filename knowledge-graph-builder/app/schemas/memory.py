"""
Memory API Schemas

Pydantic models for the Agent Memory API — store, retrieve, and forget
typed memories scoped to the Oraclous knowledge graph.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ==================== ENUMS ====================

class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryScope(str, Enum):
    SESSION = "session"
    USER = "user"
    AGENT = "agent"
    TEAM = "team"
    ORGANIZATION = "organization"


class MemorySource(str, Enum):
    AGENT = "agent"
    USER_FEEDBACK = "user_feedback"
    INGESTION = "ingestion"
    INFERENCE = "inference"


class ContradictionResolution(str, Enum):
    NEW_WINS = "new_wins"
    OLD_WINS = "old_wins"
    UNRESOLVED = "unresolved"
    MERGED = "merged"


class MemoryRetrieverType(str, Enum):
    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"
    GRAPH_TRAVERSAL = "graph_traversal"


class TemporalFilter(str, Enum):
    CURRENT = "current"
    ALL = "all"


# ==================== REQUEST SCHEMAS ====================

class MemoryCreate(BaseModel):
    """Request body for POST /graphs/{graphId}/memories"""
    type: MemoryType
    content: str = Field(..., min_length=1, max_length=10000)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    scope: MemoryScope = MemoryScope.AGENT
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    source: MemorySource = MemorySource.AGENT
    valid_from: Optional[datetime] = None

    # Semantic-specific
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    is_negation: bool = False

    # Episodic-specific
    event_type: Optional[str] = None
    user_id: Optional[str] = None

    # Procedural-specific
    category: Optional[str] = None
    trigger_pattern: Optional[str] = None


class MemoryUpdate(BaseModel):
    """Request body for PATCH /graphs/{graphId}/memories/{memoryId}"""
    content: Optional[str] = Field(default=None, min_length=1, max_length=10000)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reason: Optional[str] = None


# ==================== RESPONSE SCHEMAS ====================

class ConflictInfo(BaseModel):
    conflict_memory_id: str
    content: str
    resolution: ContradictionResolution


class MemoryCreateResponse(BaseModel):
    """Response for POST /graphs/{graphId}/memories"""
    memory_id: str
    importance_score: float
    contradictions_detected: List[ConflictInfo] = []
    entity_linked: Optional[str] = None


class MemorySearchResult(BaseModel):
    """Single memory result in search response"""
    memory_id: str
    type: MemoryType
    content: str
    importance_score: float
    relevance_score: float
    confidence: float
    valid_from: Optional[datetime]
    valid_to: Optional[datetime]
    scope: MemoryScope
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0


class GraphFact(BaseModel):
    """Entity graph fact returned alongside memories"""
    subject: str
    predicate: str
    object: str
    source_chunk_id: Optional[str] = None


class MemorySearchResponse(BaseModel):
    """Response for GET /graphs/{graphId}/memories/search"""
    memories: List[MemorySearchResult]
    graph_facts: List[GraphFact] = []
    total: int


class MemoryContext(BaseModel):
    """Response for GET /graphs/{graphId}/memories/context"""
    context_block: str
    memories_used: List[str]
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
