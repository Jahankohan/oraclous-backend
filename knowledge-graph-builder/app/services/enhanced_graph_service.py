from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from langchain_community.graphs.graph_document import GraphDocument
from app.models.graph import KnowledgeGraph, IngestionJob
from app.core.neo4j_client import neo4j_client
from app.services.schema_service import schema_service
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service
from app.core.logging import get_logger
import json
import uuid

logger = get_logger(__name__)

class EnhancedGraphService:
    """Enhanced graph service with vector capabilities"""
    
    def __init__(self):
        pass
    
    async def store_graph_documents_with_embeddings(
        self,
        graph_id: UUID,
        graph_documents: List[GraphDocument],
        user_id: str,
        generate_embeddings: bool = True
    ) -> Tuple[int, int]:
        """Store graph documents with optional embedding generation"""
        
        try:
            entities_count = 0
            relationships_count = 0
            
            # Initialize embeddings if requested
            if generate_embeddings:
                embedding_initialized = await embedding_service.initialize_embeddings(
                    provider="openai", user_id=user_id
                )
                
                if embedding_initialized:
                    # Create vector indexes with correct dimension
                    await vector_service.create_vector_indexes(
                        dimension=embedding_service.dimension
                    )
            
            for graph_doc in graph_documents:
                # Store nodes with embeddings
                for node in graph_doc.nodes:
                    await self._store_node_with_embedding(
                        node, str(graph_id), generate_embeddings
                    )
                    entities_count += 1
                
                # Store relationships  
                for rel in graph_doc.relationships:
                    await self._store_relationship(rel, str(graph_id))
                    relationships_count += 1
                
                # Create text chunks from source document if available
                if hasattr(graph_doc, 'source') and graph_doc.source:
                    await self._create_text_chunks_with_embeddings(
                        graph_doc.source, graph_id, generate_embeddings
                    )
            
            logger.info(f"Stored {entities_count} entities and {relationships_count} relationships with embeddings")
            return entities_count, relationships_count
            
        except Exception as e:
            logger.error(f"Error storing graph documents with embeddings: {e}")
            raise
    
    async def _store_node_with_embedding(
        self, 
        node, 
        graph_id: str, 
        generate_embedding: bool = True
    ):
        """Store node with optional embedding generation"""
        
        # Prepare node properties
        properties = dict(node.properties) if hasattr(node, 'properties') and node.properties else {}
        properties["graph_id"] = graph_id
        properties["id"] = node.id
        
        # Generate embedding if enabled
        embedding = None
        if generate_embedding and embedding_service.is_initialized():
            try:
                # Create text for embedding from name and description
                text_for_embedding = properties.get("name", "")
                if properties.get("description"):
                    text_for_embedding += f" {properties['description']}"
                
                if text_for_embedding.strip():
                    embedding = await embedding_service.embed_text(text_for_embedding)
                    properties["embedding"] = embedding
                    
            except Exception as e:
                logger.warning(f"Failed to generate embedding for node {node.id}: {e}")
        
        # Sanitize labels
        if isinstance(node.type, list):
            labels = [self._sanitize_label(label) for label in node.type]
        else:
            labels = [self._sanitize_label(node.type)]
        
        labels = [label for label in labels if label]
        if not labels:
            labels = ["Entity"]
        
        labels_str = ":".join(labels)
        
        # Store node
        query = f"""
        MERGE (n:{labels_str} {{id: $id, graph_id: $graph_id}})
        SET n += $properties
        RETURN n
        """
        
        await neo4j_client.execute_write_query(query, {
            "id": node.id,
            "graph_id": graph_id,
            "properties": properties
        })
    
    async def _store_relationship(self, rel, graph_id: str):
        """Store relationship (unchanged from original)"""
        
        properties = dict(rel.properties) if hasattr(rel, 'properties') and rel.properties else {}
        properties["graph_id"] = graph_id
        
        rel_type = self._sanitize_label(rel.type) if rel.type else "RELATED_TO"
        
        query = f"""
        MATCH (source {{id: $source_id, graph_id: $graph_id}})
        MATCH (target {{id: $target_id, graph_id: $graph_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        RETURN r
        """
        
        try:
            await neo4j_client.execute_write_query(query, {
                "source_id": rel.source.id,
                "target_id": rel.target.id,
                "graph_id": graph_id,
                "properties": properties
            })
        except Exception as e:
            logger.error(f"Failed to store relationship {rel.type}: {e}")
    
    async def _create_text_chunks_with_embeddings(
        self,
        source_document,
        graph_id: UUID,
        generate_embeddings: bool = True
    ):
        """Create text chunks from source document with embeddings"""
        
        if not hasattr(source_document, 'page_content'):
            return
        
        try:
            # Simple chunking strategy
            text = source_document.page_content
            chunk_size = 500
            chunk_overlap = 50
            
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk_text = text[i:i + chunk_size]
                
                chunk_id = f"chunk_{graph_id}_{i}"
                chunk_data = {
                    "id": chunk_id,
                    "text": chunk_text,
                    "source": "document_ingestion",
                    "chunk_index": i // (chunk_size - chunk_overlap)
                }
                
                # Generate embedding if enabled
                if generate_embeddings and embedding_service.is_initialized():
                    try:
                        embedding = await embedding_service.embed_text(chunk_text)
                        chunk_data["embedding"] = embedding
                    except Exception as e:
                        logger.warning(f"Failed to generate chunk embedding: {e}")
                
                chunks.append(chunk_data)
            
            # Store chunks in Neo4j
            if chunks:
                await vector_service.create_text_chunks(graph_id, chunks)
                logger.info(f"Created {len(chunks)} text chunks with embeddings")
                
        except Exception as e:
            logger.error(f"Failed to create text chunks: {e}")
    
    def _sanitize_label(self, label: str) -> str:
        """Sanitize Neo4j labels (unchanged from original)"""
        if not label:
            return ""
        
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label)
        sanitized = sanitized.strip('_')
        
        if sanitized and sanitized[0].isdigit():
        ####################### CODING AGENT STOPPED HERE ########################