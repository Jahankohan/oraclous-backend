# app/services/pipeline_service.from ..components.entity_resolver import MultiTenantEntityDeduplicatory
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

NEO4J DUAL DRIVER ARCHITECTURE:
- Uses neo4j_client.sync_driver for GraphRAG components (VectorRetriever, Neo4jWriter, etc.)
- Uses neo4j_client.execute_query() for async database operations
- Automatic driver management and connection isolation
"""

import asyncio
from typing import Dict, List, Any, Optional
from uuid import UUID
from datetime import datetime

from fastapi import BackgroundTasks, HTTPException, status

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.types import DocumentInfo, Neo4jGraph

from app.components.multi_tenant_components import MultiTenantKGWriter, create_multi_tenant_kg_writer
from app.components.entity_resolver import MultiTenantEntityDeduplicator
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
        self.neo4j_user = settings.NEO4J_USERNAME  
        self.neo4j_password = settings.NEO4J_PASSWORD
        self.neo4j_database = settings.NEO4J_DATABASE
        
        # OpenAI Configuration
        self.openai_api_key = settings.OPENAI_API_KEY
        self.llm_model = getattr(settings, 'LLM_MODEL', 'gpt-4o')  # Use gpt-4o which supports json_object
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
        
        # TODO: Implement Schema-Guided Extraction similar to benchmark's AdvancedSchemaManager
        # The benchmark implementation uses sophisticated schema learning from text samples
        # which could improve entity extraction accuracy, type consistency, and entity count
        # See benchmark.py AdvancedSchemaManager class for reference implementation


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
    
    def _model_supports_json_object(self, model_name: str) -> bool:
        """
        Check if the OpenAI model supports response_format with json_object.
        Only newer models support this feature.
        """
        json_supported_models = [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125"
        ]
        
        # Check if the model name contains any of the supported model identifiers
        return any(supported_model in model_name for supported_model in json_supported_models)
    
    async def _initialize_components(self):
        """
        Initialize Neo4j GraphRAG components using dual driver architecture.
        
        Uses sync driver for GraphRAG components and async operations through neo4j_client.
        """
        if self._initialized:
            return
        
        try:
            # Ensure both drivers are available
            await neo4j_client.connect_async()  # For async operations
            neo4j_client.connect_sync()          # For GraphRAG components
            
            # Use sync driver for GraphRAG components (required by neo4j_graphrag)
            self.driver = neo4j_client.sync_driver
            if not self.driver:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Neo4j sync connection not available for GraphRAG"
                )
            
            # Initialize OpenAI LLM with conditional response format
            model_params: Dict[str, Any] = {
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens
            }
            
            # Only add response_format for models that support it
            if self._model_supports_json_object(self.config.llm_model):
                model_params["response_format"] = {"type": "json_object"}
                logger.info(f"Using JSON object response format for model {self.config.llm_model}")
            else:
                logger.warning(f"Model {self.config.llm_model} does not support JSON object response format")
            
            self.llm = OpenAILLM(
                model_name=self.config.llm_model,
                api_key=self.config.openai_api_key,
                model_params=model_params
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
        
        This processes chunks independently which can create duplicate entities.
        The solution is to add proper entity resolution after extraction.
        """
        from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
        from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor, OnError
        from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
        
        # 0. Create DocumentInfo for proper lexical graph support
        document_info = DocumentInfo(path=source)
        
        # 1. Text Splitting
        splitter = FixedSizeSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        chunks = await splitter.run(text=text)
        
        # 2. Chunk Embedding  
        if self.embedder:
            chunk_embedder = TextChunkEmbedder(embedder=self.embedder)
            embedded_chunks = await chunk_embedder.run(text_chunks=chunks)
        else:
            embedded_chunks = chunks
        
        # 3. Entity & Relationship Extraction with document info
        if self.llm:
            extractor = LLMEntityRelationExtractor(
                llm=self.llm,
                create_lexical_graph=True,  # Creates Document/Chunk nodes
                on_error=OnError.IGNORE
            )
            graph = await extractor.run(chunks=embedded_chunks, document_info=document_info)
            
            # TODO: Add Schema-Guided Extraction similar to benchmark implementation
            # The benchmark uses AdvancedSchemaManager with schema learning from text samples
            # This could improve entity extraction accuracy and type consistency
            
            # Detailed logging for entity analysis
            logger.info(f"Raw extraction: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")
            
            # Log entity breakdown by type/label
            entity_types = {}
            entity_names = []
            for node in graph.nodes:
                node_label = getattr(node, 'label', 'Unknown')
                entity_types[node_label] = entity_types.get(node_label, 0) + 1
                if hasattr(node, 'properties') and node.properties and node.properties.get('name'):
                    entity_names.append(node.properties['name'])
            
            logger.info(f"Raw extraction entity breakdown by type: {entity_types}")
            logger.info(f"Extracted entity names: {entity_names}")
            
        else:
            logger.error("LLM not available for entity extraction")
            return {"entities_created": 0, "relationships_created": 0, "chunks_created": 0}
        
        # 4. Pre-process: Normalize entity IDs to handle chunk overlap
        logger.info("Starting entity normalization to handle chunk overlaps...")
        graph = await self._normalize_overlapping_entities(graph)
        logger.info(f"After normalization: {len(graph.nodes)} nodes, {len(graph.relationships)} relationships")
        
        # 5. Multi-tenant metadata injection (automatic via kg_writer)
        await kg_writer.run(graph)
        logger.info(f"Graph writing completed, starting entity deduplication for graph {self.graph_id}")
        
        # 6. Entity Deduplication - Consolidate duplicate entities across chunks
        logger.info(f"Checking driver availability for deduplication: driver={self.driver is not None}")
        if self.driver:  # Ensure driver is available
            logger.info(f"Creating entity deduplicator for graph {self.graph_id}")
            entity_deduplicator = MultiTenantEntityDeduplicator(
                driver=self.driver,
                graph_id=self.graph_id,
                similarity_threshold=0.85,
                enable_fuzzy_matching=False  # Start with exact matching only
            )
            
            # Run entity deduplication on the graph
            logger.info(f"Running entity deduplication for graph {self.graph_id}")
            await entity_deduplicator.run(graph)
            logger.info(f"Entity deduplication completed for graph {self.graph_id}")
        else:
            logger.warning("Neo4j driver not available - skipping entity deduplication")

        # Return statistics
        return {
            "entities_created": len(graph.nodes) if graph and graph.nodes else 0,
            "relationships_created": len(graph.relationships) if graph and graph.relationships else 0,
            "chunks_created": len(chunks.chunks) if chunks and hasattr(chunks, 'chunks') else 0
        }
    
    async def _normalize_overlapping_entities(self, graph: Neo4jGraph) -> Neo4jGraph:
        """
        Normalize entity IDs to handle chunk overlap by removing chunk prefixes
        and creating consistent entity identifiers based on entity names.
        
        This solves the chunk overlap issue where the same entity appears in multiple
        chunks with different IDs (e.g., chunk_1:Alex Thompson vs chunk_2:Alex Thompson).
        """
        from typing import Dict, Any
        
        if not graph or not graph.nodes:
            return graph
        
        # Step 1: Create mapping from entity names to canonical IDs
        entity_name_to_canonical_id: Dict[str, str] = {}
        old_id_to_new_id: Dict[str, str] = {}
        original_entity_count = len(graph.nodes)
        
        # Track what we're merging for debugging
        entities_by_name = {}
        
        for node in graph.nodes:
            if not hasattr(node, 'properties') or not node.properties:
                continue
                
            entity_name = node.properties.get('name')
            if not entity_name:
                continue
            
            # Track entities with same name for debugging
            if entity_name not in entities_by_name:
                entities_by_name[entity_name] = []
            entities_by_name[entity_name].append({
                'id': node.id,
                'label': getattr(node, 'label', 'Unknown'),
                'properties': node.properties
            })
            
            # Create canonical ID from entity name (remove chunk prefix if exists)
            original_id = node.id
            if ':' in original_id:
                # Extract the actual entity name part after chunk prefix
                canonical_id = original_id.split(':', 1)[1]
            else:
                canonical_id = original_id
            
            # Use entity name as the canonical identifier
            if entity_name not in entity_name_to_canonical_id:
                entity_name_to_canonical_id[entity_name] = canonical_id
            
            # Map old chunk-prefixed ID to canonical ID
            old_id_to_new_id[original_id] = entity_name_to_canonical_id[entity_name]
        
        # Log what we're about to merge
        entities_to_merge = {name: entities for name, entities in entities_by_name.items() if len(entities) > 1}
        if entities_to_merge:
            logger.info(f"🔄 Entities being merged due to same name:")
            for entity_name, entity_variations in entities_to_merge.items():
                logger.info(f"  📍 '{entity_name}': {len(entity_variations)} variations")
                for i, variation in enumerate(entity_variations):
                    logger.info(f"    {i+1}. ID: {variation['id']}, Label: {variation['label']}")
        else:
            logger.info("✅ No duplicate entity names found - no merging needed")
        
        # Step 2: Update node IDs to use canonical IDs
        for node in graph.nodes:
            if node.id in old_id_to_new_id:
                node.id = old_id_to_new_id[node.id]
        
        # Step 3: Update relationship references to use canonical IDs
        for rel in graph.relationships:
            if rel.start_node_id in old_id_to_new_id:
                rel.start_node_id = old_id_to_new_id[rel.start_node_id]
            if rel.end_node_id in old_id_to_new_id:
                rel.end_node_id = old_id_to_new_id[rel.end_node_id]
        
        # Step 4: Remove duplicate nodes with same canonical ID
        unique_nodes = {}
        for node in graph.nodes:
            node_key = node.id
            if node_key not in unique_nodes:
                unique_nodes[node_key] = node
            else:
                # Merge properties if needed (take the one with more properties)
                existing_node = unique_nodes[node_key]
                if (hasattr(node, 'properties') and node.properties and 
                    len(node.properties) > len(existing_node.properties or {})):
                    unique_nodes[node_key] = node
        
        # Update graph with deduplicated nodes
        graph.nodes = list(unique_nodes.values())
        
        logger.info(f"Entity normalization: Reduced {len(old_id_to_new_id)} entity references "
                   f"to {len(unique_nodes)} unique entities")
        logger.info(f"📊 Normalization summary: {original_entity_count} → {len(unique_nodes)} entities "
                   f"({original_entity_count - len(unique_nodes)} merged)")
        
        return graph
    
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
        self._pipeline_cache: Dict[str, MultiTenantGraphRAGPipeline] = {}  # Cache pipelines per graph_id
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