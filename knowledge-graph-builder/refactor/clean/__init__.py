"""Clean GraphRAG pipeline package - maintaining ALL original functionality."""

# Import all components for easy access
__version__ = "1.0.0"

from components import EntityEmbedder, LoggingText2CypherRetriever, RelationshipEmbedder
from config import AdvancedPipelineConfig, PerformanceMetrics
from document_processor import AdvancedDocumentProcessor
from entity_resolver import MultiAlgorithmEntityResolver, SimpleEntityResolver
from indexes import IndexManager
from pipeline import AdvancedGraphRAGPipeline
from retrieval import RetrievalSystemFactory
from schema_manager import AdvancedSchemaManager

__all__ = [
    "AdvancedPipelineConfig",
    "PerformanceMetrics",
    "AdvancedGraphRAGPipeline",
    "EntityEmbedder",
    "RelationshipEmbedder",
    "LoggingText2CypherRetriever",
    "SimpleEntityResolver",
    "MultiAlgorithmEntityResolver",
    "IndexManager",
    "RetrievalSystemFactory",
    "AdvancedDocumentProcessor",
    "AdvancedSchemaManager",
]
