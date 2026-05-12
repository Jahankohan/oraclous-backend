"""Configuration management for the GraphRAG pipeline."""

import os
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdvancedPipelineConfig:
    """Comprehensive configuration for advanced GraphRAG pipeline - exact same as original"""

    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # LLM Configuration
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 3000

    # Embedding Configuration
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # Chunking Configuration
    chunk_size: int = 1500
    chunk_overlap: int = 300
    approximate_chunking: bool = True

    # Processing Configuration
    batch_size: int = 2000
    max_concurrency: int = 10

    # Entity Resolution Configuration
    enable_entity_resolution: bool = True
    similarity_threshold: float = 0.85
    fuzzy_threshold: float = 0.8

    # Schema Configuration
    enable_schema_learning: bool = True
    enforce_schema: bool = True
    additional_properties_allowed: bool = False

    # Performance Configuration
    enable_performance_monitoring: bool = True
    benchmark_mode: bool = False

    # Error Handling
    on_error: str = "RAISE"  # "RAISE" or "IGNORE"
    enable_detailed_logging: bool = True


@dataclass
class PerformanceMetrics:
    """Performance monitoring and benchmarking metrics - exact same as original"""

    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    # Document processing metrics
    documents_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0

    # Entity extraction metrics
    entities_extracted: int = 0
    relationships_extracted: int = 0
    entities_resolved: int = 0

    # Database metrics
    nodes_created: int = 0
    relationships_created: int = 0
    indexes_created: int = 0

    # Performance metrics
    processing_times: list[float] = field(default_factory=list)
    memory_usage: list[float] = field(default_factory=list)

    def add_processing_time(self, duration: float):
        """Add processing time measurement"""
        self.processing_times.append(duration)

    def finalize(self):
        """Finalize metrics collection"""
        self.end_time = time.time()

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary"""
        import statistics

        total_duration = (self.end_time or time.time()) - self.start_time

        return {
            "total_duration_seconds": total_duration,
            "documents_processed": self.documents_processed,
            "chunks_created": self.chunks_created,
            "embeddings_generated": self.embeddings_generated,
            "entities_extracted": self.entities_extracted,
            "relationships_extracted": self.relationships_extracted,
            "entities_resolved": self.entities_resolved,
            "nodes_created": self.nodes_created,
            "relationships_created": self.relationships_created,
            "processing_rate_docs_per_second": self.documents_processed / total_duration
            if total_duration > 0
            else 0,
            "average_processing_time": statistics.mean(self.processing_times)
            if self.processing_times
            else 0,
            "median_processing_time": statistics.median(self.processing_times)
            if self.processing_times
            else 0,
            "p95_processing_time": (
                sorted(self.processing_times)[int(0.95 * len(self.processing_times))]
                if self.processing_times
                else 0
            ),
        }
