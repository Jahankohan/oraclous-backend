"""
Retriever Configuration Schemas

Comprehensive Pydantic schemas for all Neo4j GraphRAG retriever types.
Supports dynamic configuration and validation for multi-tenant usage.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class RetrieverType(StrEnum):
    """Supported retriever types"""

    VECTOR = "vector"
    VECTOR_CYPHER = "vector_cypher"
    HYBRID = "hybrid"
    HYBRID_CYPHER = "hybrid_cypher"
    TEXT2CYPHER = "text2cypher"
    MEMORY = "memory"


class HybridSearchRanker(StrEnum):
    """Hybrid search ranking algorithms"""

    NAIVE = "naive"
    LINEAR = "linear"


class BaseRetrieverConfig(BaseModel):
    """Base configuration for all retrievers"""

    top_k: int = Field(
        default=5, ge=1, le=100, description="Number of results to return"
    )
    effective_search_ratio: int = Field(
        default=1, ge=1, le=10, description="Controls candidate pool size"
    )
    neo4j_database: str | None = Field(default=None, description="Neo4j database name")
    return_properties: list[str] | None = Field(
        default=None, description="Node properties to return"
    )


class VectorRetrieverConfig(BaseRetrieverConfig):
    """Configuration for VectorRetriever"""

    index_name: str = Field(..., description="Vector index name")
    filters: dict[str, Any] | None = Field(
        default=None, description="Metadata pre-filtering"
    )

    class Config:
        schema_extra = {
            "example": {
                "index_name": "text_embeddings_primary",
                "top_k": 5,
                "effective_search_ratio": 1,
                "filters": {"document_type": "research_paper"},
            }
        }


class VectorCypherRetrieverConfig(BaseRetrieverConfig):
    """Configuration for VectorCypherRetriever"""

    index_name: str = Field(..., description="Vector index name")
    retrieval_query: str = Field(..., description="Cypher query for graph traversal")
    query_params: dict[str, Any] | None = Field(
        default=None, description="Parameters for Cypher query"
    )
    filters: dict[str, Any] | None = Field(
        default=None, description="Metadata pre-filtering"
    )

    class Config:
        schema_extra = {
            "example": {
                "index_name": "text_embeddings_primary",
                "retrieval_query": "MATCH (entity)-[:FROM_CHUNK]->(node) RETURN entity, node, score",
                "top_k": 5,
                "query_params": {"graph_id": "8efbff79-5675-4923-8680-34e4864bf150"},
            }
        }


class HybridRetrieverConfig(BaseRetrieverConfig):
    """Configuration for HybridRetriever"""

    vector_index_name: str = Field(..., description="Vector index name")
    fulltext_index_name: str = Field(..., description="Full-text index name")
    ranker: HybridSearchRanker = Field(
        default=HybridSearchRanker.NAIVE, description="Ranking algorithm"
    )
    alpha: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Weight for linear ranker"
    )

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float | None, info: ValidationInfo) -> float | None:
        if info.data.get("ranker") == HybridSearchRanker.LINEAR and v is None:
            raise ValueError("alpha is required when using linear ranker")
        return v

    class Config:
        schema_extra = {
            "example": {
                "vector_index_name": "text_embeddings_primary",
                "fulltext_index_name": "fulltext_chunks",
                "top_k": 5,
                "ranker": "naive",
            }
        }


class HybridCypherRetrieverConfig(BaseRetrieverConfig):
    """Configuration for HybridCypherRetriever"""

    vector_index_name: str = Field(..., description="Vector index name")
    fulltext_index_name: str = Field(..., description="Full-text index name")
    retrieval_query: str = Field(..., description="Cypher query for graph traversal")
    query_params: dict[str, Any] | None = Field(
        default=None, description="Parameters for Cypher query"
    )
    ranker: HybridSearchRanker = Field(
        default=HybridSearchRanker.NAIVE, description="Ranking algorithm"
    )
    alpha: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Weight for linear ranker"
    )

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float | None, info: ValidationInfo) -> float | None:
        if info.data.get("ranker") == HybridSearchRanker.LINEAR and v is None:
            raise ValueError("alpha is required when using linear ranker")
        return v

    class Config:
        schema_extra = {
            "example": {
                "vector_index_name": "text_embeddings_primary",
                "fulltext_index_name": "fulltext_chunks",
                "retrieval_query": "MATCH (entity)-[:FROM_CHUNK]->(node) RETURN entity, node, score",
                "top_k": 5,
                "ranker": "naive",
            }
        }


class Text2CypherRetrieverConfig(BaseRetrieverConfig):
    """Configuration for Text2CypherRetriever"""

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
                "neo4j_schema": "(:Entity)-[:RELATION]->(:Entity), (:Chunk)-[:FROM_DOCUMENT]->(:Document)",
                "examples": [
                    "Find all entities related to TechNova: MATCH (e:Entity {name: 'TechNova'})-[r]-(related) RETURN e, r, related"
                ],
                "top_k": 10,
            }
        }


class MemoryRetrieverConfig(BaseRetrieverConfig):
    """Configuration for MemoryRetriever"""

    query: str = Field(..., description="Natural language query for memory recall")
    scope: str | None = Field(default=None, description="Memory scope filter")
    memory_type: str | None = Field(default=None, description="Filter by memory type")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2000, ge=100, le=8000)


class RetrieverConfig(BaseModel):
    """Unified retriever configuration"""

    type: RetrieverType = Field(..., description="Type of retriever to use")
    config: (
        VectorRetrieverConfig
        | VectorCypherRetrieverConfig
        | HybridRetrieverConfig
        | HybridCypherRetrieverConfig
        | Text2CypherRetrieverConfig
        | MemoryRetrieverConfig
    ) = Field(..., description="Type-specific configuration")

    class Config:
        schema_extra = {
            "example": {
                "type": "vector_cypher",
                "config": {
                    "index_name": "text_embeddings_primary",
                    "retrieval_query": "MATCH (entity)-[:FROM_CHUNK]->(node) RETURN entity, node, score",
                    "top_k": 5,
                },
            }
        }


# ==================== DEFAULT CONFIGURATIONS ====================


class DefaultRetrieverConfigs:
    """Default configurations for different retriever types"""

    @staticmethod
    def get_vector_config(graph_id: str) -> VectorRetrieverConfig:
        """Get default VectorRetriever configuration"""
        return VectorRetrieverConfig(
            index_name="text_embeddings_primary", top_k=5, effective_search_ratio=1
        )

    @staticmethod
    def get_vector_cypher_config(graph_id: str) -> VectorCypherRetrieverConfig:
        """Get default VectorCypherRetriever configuration with multi-tenant support"""
        return VectorCypherRetrieverConfig(
            index_name="text_embeddings_primary",
            retrieval_query=f"""
            // Multi-tenant filter for graph_id
            WHERE node.graph_id = '{graph_id}'
            
            // Get entities that are connected to this chunk
            MATCH (entity {{graph_id: '{graph_id}'}})-[:FROM_CHUNK]->(node)
            OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(document {{graph_id: '{graph_id}'}})
            
            // Traverse entity relationships for context
            OPTIONAL MATCH (entity)-[r]-(related_entity {{graph_id: '{graph_id}'}})
            
            RETURN node.text as text,
                   document.path as document_path,
                   collect(DISTINCT entity.name) as entities,
                   collect(DISTINCT {{
                       entity: related_entity.name,
                       relationship: type(r)
                   }}) as relationships,
                   score
            ORDER BY score DESC
            """,
            top_k=5,
            effective_search_ratio=1,
        )

    @staticmethod
    def get_hybrid_config(graph_id: str) -> HybridRetrieverConfig:
        """Get default HybridRetriever configuration"""
        return HybridRetrieverConfig(
            vector_index_name="text_embeddings_primary",
            fulltext_index_name="fulltext_chunks",
            top_k=5,
            ranker=HybridSearchRanker.NAIVE,
        )

    @staticmethod
    def get_hybrid_cypher_config(graph_id: str) -> HybridCypherRetrieverConfig:
        """Get default HybridCypherRetriever configuration with multi-tenant support"""
        return HybridCypherRetrieverConfig(
            vector_index_name="text_embeddings_primary",
            fulltext_index_name="fulltext_chunks",
            retrieval_query=f"""
            // Multi-tenant filter for graph_id
            WHERE node.graph_id = '{graph_id}'
            
            // Get entities that are connected to this chunk
            MATCH (entity {{graph_id: '{graph_id}'}})-[:FROM_CHUNK]->(node)
            OPTIONAL MATCH (node)-[:FROM_DOCUMENT]->(document {{graph_id: '{graph_id}'}})
            
            // Traverse entity relationships for context
            OPTIONAL MATCH (entity)-[r]-(related_entity {{graph_id: '{graph_id}'}})
            
            RETURN node.text as text,
                   document.path as document_path,
                   collect(DISTINCT entity.name) as entities,
                   collect(DISTINCT {{
                       entity: related_entity.name,
                       relationship: type(r)
                   }}) as relationships,
                   score
            ORDER BY score DESC
            """,
            top_k=5,
            ranker=HybridSearchRanker.NAIVE,
        )

    @staticmethod
    def get_text2cypher_config(graph_id: str) -> Text2CypherRetrieverConfig:
        """Get default Text2CypherRetriever configuration"""
        return Text2CypherRetrieverConfig(
            neo4j_schema=f"""
            // Multi-tenant Knowledge Graph Schema for graph_id: {graph_id}
            
            // Core entity types
            (:Entity {{name: string, type: string, graph_id: string}})
            (:Chunk {{text: string, graph_id: string, embedding: vector}})
            (:Document {{path: string, title: string, graph_id: string}})
            
            // Entity relationships
            (:Entity)-[:FOUNDED_BY]->(:Entity)
            (:Entity)-[:CEO_OF]->(:Entity)
            (:Entity)-[:LOCATED_IN]->(:Entity)
            (:Entity)-[:PARTNER_WITH]->(:Entity)
            (:Entity)-[:DEVELOPS]->(:Entity)
            
            // Document relationships
            (:Entity)-[:FROM_CHUNK]->(:Chunk)
            (:Chunk)-[:FROM_DOCUMENT]->(:Document)
            
            // All nodes have graph_id property for multi-tenant isolation
            """,
            examples=[
                f"Find all entities in this graph: MATCH (e:Entity {{graph_id: '{graph_id}'}}) RETURN e LIMIT 10",
                f"Find entities related to a specific entity: MATCH (e:Entity {{graph_id: '{graph_id}', name: 'TechNova Corporation'}})-[r]-(related) RETURN e, r, related",
                f"Find documents containing specific information: MATCH (d:Document {{graph_id: '{graph_id}'}})<-[:FROM_DOCUMENT]-(c:Chunk) WHERE c.text CONTAINS 'innovation' RETURN d, c",
            ],
            top_k=10,
        )


# ==================== UTILITY FUNCTIONS ====================


def get_default_retriever_config(
    retriever_type: RetrieverType, graph_id: str
) -> BaseRetrieverConfig:
    """Get default configuration for a specific retriever type"""
    config_map = {
        RetrieverType.VECTOR: DefaultRetrieverConfigs.get_vector_config,
        RetrieverType.VECTOR_CYPHER: DefaultRetrieverConfigs.get_vector_cypher_config,
        RetrieverType.HYBRID: DefaultRetrieverConfigs.get_hybrid_config,
        RetrieverType.HYBRID_CYPHER: DefaultRetrieverConfigs.get_hybrid_cypher_config,
        RetrieverType.TEXT2CYPHER: DefaultRetrieverConfigs.get_text2cypher_config,
        RetrieverType.MEMORY: lambda gid: MemoryRetrieverConfig(query="", top_k=20),
    }

    return config_map[retriever_type](graph_id)


def validate_retriever_config(config: RetrieverConfig) -> bool:
    """Validate retriever configuration"""
    try:
        # Pydantic validation handles most cases
        config.model_dump()
        return True
    except Exception:
        return False
