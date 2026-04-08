"""
Chat API Schemas with Enhanced Retriever Support

Comprehensive Pydantic schemas for chat requests and responses supporting
all Neo4j GraphRAG retriever types with proper validation and examples.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.graph_schemas import TemporalFilter
from app.services.retriever_factory import RetrieverType


class ChatMode(StrEnum):
    """Available chat modes (aliases for retriever types)"""

    SIMPLE = "simple"  # Vector similarity search
    ENHANCED = "enhanced"  # Vector search with graph traversal (default)
    HYBRID = "hybrid"  # Vector + full-text search
    HYBRID_PLUS = "hybrid_plus"  # Hybrid search with graph traversal
    NATURAL = "natural"  # Natural language to Cypher


class RetrieverConfigRequest(BaseModel):
    """Configuration for specific retriever types in API requests"""

    top_k: int | None = Field(
        default=5, ge=1, le=100, description="Number of results to retrieve"
    )
    effective_search_ratio: int | None = Field(
        default=1, ge=1, le=10, description="Search ratio for vector/hybrid retrievers"
    )

    # Vector-specific
    index_name: str | None = Field(
        default=None, description="Vector index name (auto-detected if not provided)"
    )

    # Cypher-specific
    retrieval_query: str | None = Field(
        default=None, description="Custom Cypher retrieval query"
    )

    # Hybrid-specific
    fulltext_index_name: str | None = Field(
        default=None, description="Full-text index name (auto-detected if not provided)"
    )
    ranker: str | None = Field(
        default="naive", description="Ranking algorithm: naive, linear"
    )
    alpha: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Weight for linear ranker"
    )

    # Text2Cypher-specific
    neo4j_schema: str | None = Field(
        default=None, description="Neo4j schema description"
    )
    examples: list[str] | None = Field(
        default=None, description="Example queries for few-shot learning"
    )
    custom_prompt: str | None = Field(
        default=None, description="Custom prompt template"
    )
    llm_params: dict[str, Any] | None = Field(
        default=None, description="LLM-specific parameters"
    )

    class Config:
        schema_extra = {
            "example": {
                "top_k": 5,
                "effective_search_ratio": 2,
                "ranker": "linear",
                "alpha": 0.7,
            }
        }


class ChatRequest(BaseModel):
    """Enhanced chat request supporting all retriever types"""

    query: str = Field(
        ..., min_length=1, max_length=10000, description="User's question or query"
    )
    graph_id: str = Field(..., description="Knowledge graph identifier")

    # Retriever selection
    mode: ChatMode | None = Field(
        default=ChatMode.ENHANCED, description="Chat mode/retriever type"
    )
    retriever_type: RetrieverType | None = Field(
        default=None, description="Explicit retriever type (overrides mode)"
    )
    retriever_config: RetrieverConfigRequest | None = Field(
        default=None, description="Retriever-specific configuration"
    )

    # Response options
    return_context: bool = Field(
        default=False, description="Include retrieval context in response"
    )
    include_sources: bool = Field(
        default=True, description="Include source information"
    )
    include_cypher: bool = Field(
        default=False, description="Include generated Cypher queries (for Text2Cypher)"
    )

    # Temporal scoping — restricts retrieved facts to a point or range in time
    temporal_filter: TemporalFilter | None = Field(
        default=None,
        description="When set, scopes retrieval to facts valid at the specified time. "
        "Clients without this field get current-only results by default.",
    )

    # Chat context
    examples: str = Field(default="", description="Examples for few-shot learning")
    conversation_id: str | None = Field(
        default=None, description="Conversation session ID"
    )

    @field_validator("retriever_config")
    @classmethod
    def validate_retriever_config(
        cls, v: RetrieverConfigRequest | None
    ) -> RetrieverConfigRequest | None:
        if v and v.ranker == "linear" and v.alpha is None:
            raise ValueError("alpha is required when using linear ranker")
        return v

    class Config:
        schema_extra = {
            "examples": [
                {
                    "query": "Tell me about TechNova Corporation",
                    "graph_id": "8efbff79-5675-4923-8680-34e4864bf150",
                    "mode": "enhanced",
                    "return_context": True,
                },
                {
                    "query": "Find all partnerships involving tech companies",
                    "graph_id": "8efbff79-5675-4923-8680-34e4864bf150",
                    "mode": "hybrid_plus",
                    "retriever_config": {"top_k": 10, "ranker": "linear", "alpha": 0.8},
                },
                {
                    "query": "What are the key entities in this knowledge graph?",
                    "graph_id": "8efbff79-5675-4923-8680-34e4864bf150",
                    "mode": "natural",
                    "include_cypher": True,
                },
            ]
        }


class SourceInfo(BaseModel):
    """Information about a source graph node or document chunk"""

    node_id: str | None = Field(default=None, description="Neo4j node element ID")
    node_labels: list[str] | None = Field(default=None, description="Neo4j node labels")
    document_path: str | None = Field(default=None, description="Source document path")
    chunk_id: str | None = Field(default=None, description="Chunk identifier")
    relevance_score: float | None = Field(default=None, description="Relevance score")
    content: str | None = Field(
        default=None, description="Excerpt of the node content used"
    )
    entities: list[str] | None = Field(default=None, description="Related entities")
    relationships: list[dict[str, str]] | None = Field(
        default=None, description="Related relationships"
    )
    properties: dict[str, Any] | None = Field(
        default=None, description="Additional node properties"
    )


class RetrievalContext(BaseModel):
    """Detailed retrieval context and metadata"""

    retriever_type: str = Field(..., description="Type of retriever used")
    sources: list[SourceInfo] = Field(default=[], description="Source information")
    cypher_queries: list[str] | None = Field(
        default=None, description="Generated Cypher queries"
    )
    vector_scores: list[float] | None = Field(
        default=None, description="Vector similarity scores"
    )
    fulltext_scores: list[float] | None = Field(
        default=None, description="Full-text search scores"
    )
    total_results: int = Field(
        default=0, description="Total number of retrieved results"
    )
    processing_time_ms: float | None = Field(
        default=None, description="Processing time in milliseconds"
    )


class ChatResponse(BaseModel):
    """Enhanced chat response with comprehensive metadata"""

    # Core response
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original user query")
    graph_id: str = Field(..., description="Knowledge graph identifier")
    success: bool = Field(..., description="Whether the request was successful")

    # Retrieval information
    mode: str = Field(..., description="Chat mode used")
    retriever_type: str = Field(..., description="Retriever type used")

    # Grounding / hallucination prevention
    is_grounded: bool = Field(
        default=True,
        description=(
            "True if the answer is fully grounded in retrieved graph data. "
            "False if the graph contained insufficient data to answer."
        ),
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score [0–1] derived from retrieval relevance scores",
    )

    # Optional detailed information
    context: RetrievalContext | None = Field(
        default=None, description="Detailed retrieval context"
    )
    sources: list[SourceInfo] | None = Field(
        default=None, description="Source information"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Response timestamp"
    )
    conversation_id: str | None = Field(
        default=None, description="Conversation session ID"
    )

    class Config:
        schema_extra = {
            "example": {
                "answer": "TechNova Corporation is a technology company founded in 2015...",
                "query": "Tell me about TechNova Corporation",
                "graph_id": "8efbff79-5675-4923-8680-34e4864bf150",
                "success": True,
                "mode": "enhanced",
                "retriever_type": "vector_cypher",
                "sources": [
                    {
                        "document_path": "/documents/technova_profile.pdf",
                        "relevance_score": 0.95,
                        "entities": [
                            "TechNova Corporation",
                            "Technology",
                            "Innovation",
                        ],
                    }
                ],
                "metadata": {"model": "gpt-4o", "total_tokens": 1250},
            }
        }


class ChatModeInfo(BaseModel):
    """Information about a specific chat mode"""

    mode: ChatMode = Field(..., description="Chat mode identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description")
    retriever_type: RetrieverType = Field(..., description="Underlying retriever type")
    use_cases: list[str] = Field(..., description="Recommended use cases")
    requires_fulltext_index: bool = Field(
        default=False, description="Whether full-text indexes are required"
    )

    class Config:
        schema_extra = {
            "example": {
                "mode": "enhanced",
                "name": "Enhanced Search",
                "description": "Vector similarity search combined with graph traversal for rich context",
                "retriever_type": "vector_cypher",
                "use_cases": [
                    "Finding entities with relationships",
                    "Getting comprehensive context",
                    "Exploring graph connections",
                ],
                "requires_fulltext_index": False,
            }
        }


class ChatModesResponse(BaseModel):
    """Response containing all available chat modes"""

    modes: list[ChatModeInfo] = Field(..., description="Available chat modes")
    default_mode: ChatMode = Field(..., description="Default recommended mode")
    graph_capabilities: dict[str, bool] = Field(
        default_factory=dict, description="Graph-specific capabilities"
    )

    class Config:
        schema_extra = {
            "example": {
                "modes": [
                    {
                        "mode": "simple",
                        "name": "Simple Search",
                        "description": "Basic vector similarity search",
                        "retriever_type": "vector",
                        "use_cases": ["Quick facts", "Simple questions"],
                    }
                ],
                "default_mode": "enhanced",
                "graph_capabilities": {
                    "has_fulltext_indexes": True,
                    "has_vector_indexes": True,
                    "supports_cypher": True,
                },
            }
        }


class ErrorResponse(BaseModel):
    """Standardized error response"""

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Error timestamp"
    )

    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "RETRIEVER_ERROR",
                "message": "Failed to initialize hybrid retriever: full-text index not found",
                "details": {
                    "graph_id": "8efbff79-5675-4923-8680-34e4864bf150",
                    "required_index": "fulltext_chunks",
                },
            }
        }


# ==================== HELPER FUNCTIONS ====================


def get_mode_mapping() -> dict[ChatMode, RetrieverType]:
    """Get mapping from chat modes to retriever types"""
    return {
        ChatMode.SIMPLE: RetrieverType.VECTOR,
        ChatMode.ENHANCED: RetrieverType.VECTOR_CYPHER,
        ChatMode.HYBRID: RetrieverType.HYBRID,
        ChatMode.HYBRID_PLUS: RetrieverType.HYBRID_CYPHER,
        ChatMode.NATURAL: RetrieverType.TEXT2CYPHER,
    }


def get_mode_info(mode: ChatMode) -> ChatModeInfo:
    """Get detailed information about a chat mode"""
    mode_mapping = get_mode_mapping()

    mode_configs = {
        ChatMode.SIMPLE: ChatModeInfo(
            mode=mode,
            name="Simple Search",
            description="Fast vector similarity search for quick facts and direct questions",
            retriever_type=mode_mapping[mode],
            use_cases=[
                "Quick factual questions",
                "Simple entity lookups",
                "Fast responses",
            ],
        ),
        ChatMode.ENHANCED: ChatModeInfo(
            mode=mode,
            name="Enhanced Search",
            description="Vector similarity search combined with graph traversal for comprehensive context",
            retriever_type=mode_mapping[mode],
            use_cases=[
                "Complex questions requiring context",
                "Entity relationships exploration",
                "Comprehensive analysis",
            ],
        ),
        ChatMode.HYBRID: ChatModeInfo(
            mode=mode,
            name="Hybrid Search",
            description="Combines vector similarity and full-text search for broader coverage",
            retriever_type=mode_mapping[mode],
            use_cases=[
                "Text and semantic search",
                "Keyword-based queries",
                "Broader result coverage",
            ],
            requires_fulltext_index=True,
        ),
        ChatMode.HYBRID_PLUS: ChatModeInfo(
            mode=mode,
            name="Hybrid Plus",
            description="Hybrid search with graph traversal for maximum context and coverage",
            retriever_type=mode_mapping[mode],
            use_cases=[
                "Complex analytical questions",
                "Research and exploration",
                "Maximum context retrieval",
            ],
            requires_fulltext_index=True,
        ),
        ChatMode.NATURAL: ChatModeInfo(
            mode=mode,
            name="Natural Query",
            description="Natural language to Cypher translation for precise graph queries",
            retriever_type=mode_mapping[mode],
            use_cases=[
                "Complex graph analytics",
                "Specific relationship queries",
                "Custom analysis requirements",
            ],
        ),
    }

    return mode_configs[mode]


def get_all_modes() -> list[ChatModeInfo]:
    """Get information about all available chat modes"""
    return [get_mode_info(mode) for mode in ChatMode]
