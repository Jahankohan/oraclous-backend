"""Clean GraphRAG pipeline package - maintaining ALL original functionality."""

# Import all components for easy access
__version__ = "1.0.0"

from config import AdvancedPipelineConfig, PerformanceMetrics
from pipeline import AdvancedGraphRAGPipeline
from components import EntityEmbedder, RelationshipEmbedder, LoggingText2CypherRetriever
from entity_resolver import SimpleEntityResolver, MultiAlgorithmEntityResolver
from indexes import IndexManager
from retrieval import RetrievalSystemFactory
from document_processor import AdvancedDocumentProcessor
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
    "AdvancedSchemaManager"
]
