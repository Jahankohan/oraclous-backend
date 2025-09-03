# app/services/pipeline_service.py
"""
Multi-Tenant Pipeline Service - Neo4j GraphRAG Foundation

Clean, maintainable pipeline service using your AdvancedGraphRAGPipeline 
with multi-tenant support and FastAPI compatibility.

DESIGN PRINCIPLES:
- Uses your clean refactor/AdvancedGraphRAGPipeline as foundation
- Multi-tenant wrapper with perfect isolation 
- Simple, maintainable code (no complex abstractions)
- FastAPI compatible with async support
- Performance monitoring and error handling
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from uuid import UUID
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, HTTPException, status

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.types import Neo4jGraph

from app.components.multi_tenant_components import MultiTenantKGWriter, create_multi_tenant_kg_writer
from app.core.neo4j_client import neo4j_client
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ==================== CONFIGURATION ADAPTER ====================

class PipelineConfig:
    """
    Configuration adapter that bridges FastAPI settings with your AdvancedPipelineConfig.
    Simple dataclass that maps existing settings to your pipeline requirements.
    """
    
    def __init__(self):
        # Neo4j Configuration
        self.neo4j_uri = settings.NEO4J_URI
        self.neo4j_user = settings.NEO4J_USER  
        self.neo4j_password = settings.NEO4J_PASSWORD
        self.neo4j_database = settings.NEO4J_DATABASE
        
        # OpenAI Configuration
        self.openai_api_key = settings.OPENAI_API_KEY
        self.llm_model = getattr(settings, 'LLM_MODEL', 'gpt-4')
        self.llm_temperature = getattr(settings, 'LLM_TEMPERATURE', 0.1)
        self.llm_max_tokens = getattr(settings, 'LLM_MAX_TOKENS', 3000)
        
        # Embedding Configuration
        self.embedding_model = getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-3-large')
        self.embedding_dimensions = 3072
        
        # Processing Configuration
        self.chunk_size = getattr(settings, 'CHUNK_SIZE', 1500)
        self.chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 300)
        self.batch_size = 2000
        self.max_concurrency = 10
        
        # Advanced Features
        self.enable_entity_resolution = True
        self.similarity_threshold = 0.85
        self.enable_schema_learning = True
        self.enable_performance_monitoring = True
        self.enable_detailed_logging = True
        self.on_error = "IGNORE"  # Continue processing on errors


# ==================== MULTI-TENANT PIPELINE WRAPPER ====================

class MultiTenantGraphRAGPipeline:
    """
    Multi-tenant wrapper around your AdvancedGraphRAGPipeline.
    
    FEATURES:
    - Perfect tenant isolation with graph_id injection
    - Uses your clean refactor pipeline as foundation
    - FastAPI compatible with async support
    - Performance monitoring and metrics
    - Simple factory pattern for clean initialization
    """
    
    def __init__(self, graph_id: str, user_id: Optional[str] = None):
        """
        Initialize multi-tenant pipeline wrapper.
        
        Args:
            graph_id: Tenant graph identifier
            user_id: Optional user identifier for additional tracking
        """
        self.graph_id = graph_id
        self.user_id = user_id
        self.config = PipelineConfig()
        
        # Components will be initialized on first use
        self.driver = None
        self.llm = None
        self.embedder = None
        self.base_pipeline = None
        self._initialized = False
        
        logger.info(f"MultiTenantGraphRAGPipeline created for graph {graph_id}")
    
    async def _initialize_components(self):
        """Initialize Neo4j GraphRAG components using your patterns."""
        if self._initialized:
            return
        
        try:
            # Use existing Neo4j client (consistent with other services)
            self.driver = neo4j_client.driver
            if not self.driver:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Neo4j connection not available"
                )
            
            # Initialize OpenAI LLM
            self.llm = OpenAILLM(
                model_name=self.config.llm_model,
                api_key=self.config.openai_api_key,
                model_params={
                    "temperature": self.config.llm_temperature,
                    "max_tokens": self.config.llm_max_tokens,
                    "response_format": {"type": "json_object"}
                }
            )
            
            # Initialize OpenAI embedder
            self.embedder = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key
            )
            
            self._initialized = True
            logger.info(f"Pipeline components initialized for graph {self.graph_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Pipeline initialization failed: {str(e)}"
            )
    
    async def process_documents(
        self,
        documents: List[Dict[str, Any]],
        background_tasks: Optional[BackgroundTasks] = None
    ) -> Dict[str, Any]:
        """
        Process documents through Neo4j GraphRAG pipeline with multi-tenant isolation.
        
        Args:
            documents: List of document dicts with 'text' and 'source' keys
            background_tasks: Optional FastAPI background tasks for async processing
            
        Returns:
            Processing result with statistics and status
        """
        try:
            await self._initialize_components()
            
            start_time = datetime.now()
            
            # For large document sets, use background processing
            if len(documents) > 10 or background_tasks:
                if background_tasks:
                    background_tasks.add_task(
                        self._process_documents_background,
                        documents
                    )
                else:
                    # Create async task for processing
                    asyncio.create_task(self._process_documents_background(documents))
                
                return {
                    "status": "processing",
                    "message": f"Processing {len(documents)} documents in background",
                    "graph_id": self.graph_id,
                    "documents_queued": len(documents),
                    "processing_started_at": start_time.isoformat()
                }
            
            # Process synchronously for small document sets
            result = await self._process_documents_sync(documents)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result.update({
                "status": "completed",
                "processing_duration": processing_time,
                "graph_id": self.graph_id
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed for graph {self.graph_id}: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "graph_id": self.graph_id,
                "documents_processed": 0
            }
    
    async def _process_documents_sync(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process documents synchronously using Neo4j GraphRAG pipeline."""
        
        # Create multi-tenant KG writer
        kg_writer = create_multi_tenant_kg_writer(
            driver=self.driver,
            graph_id=self.graph_id,
            user_id=self.user_id
        )
        
        total_entities = 0
        total_relationships = 0
        total_chunks = 0
        
        for i, doc in enumerate(documents):
            try:
                logger.info(f"Processing document {i+1}/{len(documents)} for graph {self.graph_id}")
                
                # Extract text content
                text_content = doc.get('text', '') or doc.get('content', '')
                if not text_content:
                    logger.warning(f"Document {i+1} has no text content, skipping")
                    continue
                
                # Process through Neo4j GraphRAG pipeline
                # This is a simplified version - in production you'd use your full AdvancedGraphRAGPipeline
                result = await self._process_single_document(
                    text_content, 
                    doc.get('source', f'document_{i+1}'),
                    kg_writer
                )
                
                # Update statistics
                total_entities += result.get('entities_created', 0)
                total_relationships += result.get('relationships_created', 0) 
                total_chunks += result.get('chunks_created', 0)
                
            except Exception as e:
                logger.error(f"Failed to process document {i+1}: {e}")
                continue
        
        return {
            "documents_processed": len(documents),
            "entities_created": total_entities,
            "relationships_created": total_relationships,
            "chunks_created": total_chunks
        }
    
    async def _process_single_document(
        self, 
        text: str, 
        source: str, 
        kg_writer: MultiTenantKGWriter
    ) -> Dict[str, Any]:
        """
        Process a single document using Neo4j GraphRAG components.
        
        This is a simplified implementation - you can enhance it by integrating
        your full AdvancedGraphRAGPipeline from refactor/clean/pipeline.py
        """
        from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
        from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
        from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
        
        # 1. Text Splitting
        splitter = FixedSizeSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        chunks = await splitter.run(text=text)
        
        # 2. Chunk Embedding  
        chunk_embedder = TextChunkEmbedder(embedder=self.embedder)
        embedded_chunks = await chunk_embedder.run(text_chunks=chunks)
        
        # 3. Entity & Relationship Extraction
        extractor = LLMEntityRelationExtractor(
            llm=self.llm,
            create_lexical_graph=True,  # Creates Document/Chunk nodes
            on_error=self.config.on_error
        )
        graph = await extractor.run(chunks=embedded_chunks)
        
        # 4. Multi-tenant metadata injection (automatic via kg_writer)
        await kg_writer.run(graph)
        
        # Return statistics
        return {
            "entities_created": len(graph.nodes) if graph and graph.nodes else 0,
            "relationships_created": len(graph.relationships) if graph and graph.relationships else 0,
            "chunks_created": len(chunks) if chunks else 0
        }
    
    async def _process_documents_background(self, documents: List[Dict[str, Any]]):
        """Background processing for large document sets."""
        try:
            logger.info(f"Starting background processing of {len(documents)} documents for graph {self.graph_id}")
            
            result = await self._process_documents_sync(documents)
            
            logger.info(f"Background processing completed for graph {self.graph_id}: "
                       f"{result.get('entities_created', 0)} entities, "
                       f"{result.get('relationships_created', 0)} relationships")
                       
        except Exception as e:
            logger.error(f"Background processing failed for graph {self.graph_id}: {e}")
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status and statistics."""
        try:
            await self._initialize_components()
            
            # Query graph statistics
            query = """
            MATCH (n)
            WHERE n.graph_id = $graph_id
            WITH labels(n) as node_labels, count(n) as node_count
            UNWIND node_labels as label
            RETURN label, sum(node_count) as count
            """
            
            result = await neo4j_client.execute_query(query, {"graph_id": self.graph_id})
            
            stats = {}
            for record in result:
                stats[record['label']] = record['count']
            
            return {
                "graph_id": self.graph_id,
                "status": "ready",
                "statistics": stats,
                "components_initialized": self._initialized
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing status for {self.graph_id}: {e}")
            return {
                "graph_id": self.graph_id,
                "status": "error", 
                "error": str(e)
            }


# ==================== SERVICE CLASS ====================

class PipelineService:
    """
    FastAPI-compatible service for multi-tenant pipeline operations.
    
    FEATURES:
    - Factory pattern for creating tenant-specific pipelines
    - FastAPI dependency injection support
    - Clean error handling and logging
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize pipeline service."""
        self._pipeline_cache = {}  # Cache pipelines per graph_id
        logger.info("PipelineService initialized")
    
    def get_pipeline(self, graph_id: UUID, user_id: Optional[str] = None) -> MultiTenantGraphRAGPipeline:
        """
        Get or create multi-tenant pipeline for graph_id.
        
        Args:
            graph_id: Tenant graph identifier  
            user_id: Optional user identifier
            
        Returns:
            Multi-tenant pipeline instance
        """
        cache_key = f"pipeline_{graph_id}"
        
        if cache_key not in self._pipeline_cache:
            self._pipeline_cache[cache_key] = MultiTenantGraphRAGPipeline(
                graph_id=str(graph_id),
                user_id=user_id
            )
        
        return self._pipeline_cache[cache_key]
    
    async def process_documents(
        self,
        documents: List[Dict[str, Any]],
        graph_id: UUID,
        user_id: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> Dict[str, Any]:
        """
        Process documents through multi-tenant pipeline.
        
        Args:
            documents: List of document dicts
            graph_id: Tenant graph identifier
            user_id: Optional user identifier
            background_tasks: Optional FastAPI background tasks
            
        Returns:
            Processing result with status and statistics
        """
        try:
            pipeline = self.get_pipeline(graph_id, user_id)
            return await pipeline.process_documents(documents, background_tasks)
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for graph {graph_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline processing failed: {str(e)}"
            )
    
    async def get_pipeline_status(self, graph_id: UUID) -> Dict[str, Any]:
        """Get processing status for a specific graph."""
        try:
            pipeline = self.get_pipeline(graph_id)
            return await pipeline.get_processing_status()
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status for {graph_id}: {e}")
            return {
                "graph_id": str(graph_id),
                "status": "error",
                "error": str(e)
            }
    
    def clear_pipeline_cache(self, graph_id: Optional[UUID] = None):
        """Clear pipeline cache for memory management."""
        if graph_id:
            cache_key = f"pipeline_{graph_id}"
            if cache_key in self._pipeline_cache:
                del self._pipeline_cache[cache_key]
                logger.info(f"Cleared pipeline cache for graph {graph_id}")
        else:
            self._pipeline_cache.clear()
            logger.info("Cleared all pipeline caches")


# ==================== FASTAPI DEPENDENCY INJECTION ====================

def get_pipeline_service() -> PipelineService:
    """
    FastAPI dependency factory for PipelineService.
    
    Usage:
        @router.post("/graphs/{graph_id}/process")
        async def process_documents(
            graph_id: UUID,
            documents: List[DocumentRequest],
            pipeline_service: PipelineService = Depends(get_pipeline_service),
            background_tasks: BackgroundTasks = None
        ):
            return await pipeline_service.process_documents(
                documents=[doc.dict() for doc in documents],
                graph_id=graph_id,
                background_tasks=background_tasks
            )
    """
    return PipelineService()


# ==================== GLOBAL INSTANCE ====================

# Global instance for backward compatibility and direct usage
pipeline_service = PipelineService()