import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
from datetime import datetime

from langchain.schema import Document
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain.schema.graph_document import GraphDocument
from langchain_community.graphs.graph_document import GraphDocument

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import ExtractionError
from app.config.settings import get_settings
from app.models.responses import ProcessingProgress, ProcessingStatus
from app.utils.llm_clients import LLMClientFactory
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class ExtractionService:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.llm_factory = LLMClientFactory()
    
    async def extract_graph(
        self, 
        file_names: List[str], 
        model: str,
        node_labels: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        enable_schema: bool = True
    ) -> AsyncGenerator[ProcessingProgress, None]:
        """Extract knowledge graph from documents"""
        
        try:
            # Get documents to process
            documents = await self._get_documents_to_process(file_names)
            
            if not documents:
                raise ExtractionError("No documents found for processing")
            
            # Initialize LLM
            llm = self.llm_factory.get_llm(model)
            
            # Create graph transformer
            graph_transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=node_labels if enable_schema and node_labels else [],
                allowed_relationships=relationship_types if enable_schema and relationship_types else []
            )
            
            # Process each document
            for i, doc_info in enumerate(documents):
                try:
                    yield ProcessingProgress(
                        file_name=doc_info["fileName"],
                        status=ProcessingStatus.PROCESSING,
                        progress_percentage=0.0,
                        chunks_processed=0,
                        total_chunks=doc_info["totalChunks"],
                        current_step="Loading chunks"
                    )
                    
                    # Get document chunks
                    chunks = await self._get_document_chunks(doc_info["id"])
                    
                    # Process chunks in batches
                    batch_size = self.settings.chunks_to_combine
                    total_batches = (len(chunks) + batch_size - 1) // batch_size
                    
                    for batch_idx in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[batch_idx:batch_idx + batch_size]
                        
                        yield ProcessingProgress(
                            file_name=doc_info["fileName"],
                            status=ProcessingStatus.PROCESSING,
                            progress_percentage=(batch_idx / len(chunks)) * 100,
                            chunks_processed=batch_idx,
                            total_chunks=len(chunks),
                            current_step=f"Extracting entities from batch {batch_idx // batch_size + 1}/{total_batches}"
                        )
                        
                        # Extract graph documents
                        graph_documents = await self._extract_batch_graph(
                            graph_transformer, 
                            batch_chunks
                        )
                        
                        # Store graph documents
                        await self._store_graph_documents(graph_documents, doc_info["id"])
                        
                        # Generate embeddings for chunks
                        if self.settings.enable_embeddings:
                            yield ProcessingProgress(
                                file_name=doc_info["fileName"],
                                status=ProcessingStatus.PROCESSING,
                                progress_percentage=(batch_idx / len(chunks)) * 100,
                                chunks_processed=batch_idx,
                                total_chunks=len(chunks),
                                current_step="Generating embeddings"
                            )
                            
                            await self._generate_chunk_embeddings(batch_chunks)
                    
                    # Mark document as completed
                    await self._update_document_status(doc_info["id"], ProcessingStatus.COMPLETED)
                    
                    yield ProcessingProgress(
                        file_name=doc_info["fileName"],
                        status=ProcessingStatus.COMPLETED,
                        progress_percentage=100.0,
                        chunks_processed=len(chunks),
                        total_chunks=len(chunks),
                        current_step="Completed"
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_info['fileName']}: {e}")
                    await self._update_document_status(doc_info["id"], ProcessingStatus.FAILED)
                    
                    yield ProcessingProgress(
                        file_name=doc_info["fileName"],
                        status=ProcessingStatus.FAILED,
                        progress_percentage=0.0,
                        chunks_processed=0,
                        total_chunks=doc_info["totalChunks"],
                        current_step="Failed",
                        error_message=str(e)
                    )
        
        except Exception as e:
            logger.error(f"Error in extraction process: {e}")
            raise ExtractionError(f"Extraction failed: {e}")
    
    async def _get_documents_to_process(self, file_names: List[str]) -> List[Dict[str, Any]]:
        """Get documents that need processing"""
        if file_names:
            query = """
            MATCH (d:Document)
            WHERE d.fileName IN $fileNames
            RETURN d.id as id, d.fileName as fileName, 
                   coalesce(d.totalChunks, 0) as totalChunks
            """
            result = self.neo4j.execute_query(query, {"fileNames": file_names})
        else:
            query = """
            MATCH (d:Document)
            WHERE d.status = $status
            RETURN d.id as id, d.fileName as fileName, 
                   coalesce(d.totalChunks, 0) as totalChunks
            """
            result = self.neo4j.execute_query(query, {"status": ProcessingStatus.NEW.value})
        
        return result
    
    async def _get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get chunks for a document"""
        query = """
        MATCH (d:Document {id: $docId})-[:HAS_CHUNK]->(c:Chunk)
        RETURN c.id as id, c.text as text, c.chunkIndex as chunkIndex
        ORDER BY c.chunkIndex
        """
        
        return self.neo4j.execute_query(query, {"docId": document_id})
    
    async def _extract_batch_graph(
        self, 
        graph_transformer: LLMGraphTransformer, 
        chunks: List[Dict[str, Any]]
    ) -> List[GraphDocument]:
        """Extract graph from a batch of chunks"""
        # Combine chunks into documents
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk["id"],
                    "chunk_index": chunk["chunkIndex"]
                }
            )
            documents.append(doc)
        
        # Extract graph documents
        graph_documents = await asyncio.to_thread(
            graph_transformer.convert_to_graph_documents,
            documents
        )
        
        return graph_documents
    
    async def _store_graph_documents(self, graph_documents: List[GraphDocument], document_id: str) -> None:
        """Store extracted graph documents in Neo4j"""
        for graph_doc in graph_documents:
            # Create entity nodes
            for node in graph_doc.nodes:
                query = """
                MERGE (e:Entity {id: $id})
                SET e += $properties
                SET e:$label
                """
                
                # Handle multiple labels
                labels = [node.type] if hasattr(node, 'type') else ['Entity']
                for label in labels:
                    self.neo4j.execute_write_query(
                        query.replace(':$label', f':`{label}`'),
                        {
                            "id": node.id,
                            "properties": dict(node.properties) if hasattr(node, 'properties') else {}
                        }
                    )
            
            # Create relationships
            for relationship in graph_doc.relationships:
                query = """
                MATCH (source:Entity {id: $sourceId})
                MATCH (target:Entity {id: $targetId})
                MERGE (source)-[r:$relType]->(target)
                SET r += $properties
                """
                
                self.neo4j.execute_write_query(
                    query.replace(':$relType', f':`{relationship.type}`'),
                    {
                        "sourceId": relationship.source.id,
                        "targetId": relationship.target.id,
                        "properties": dict(relationship.properties) if hasattr(relationship, 'properties') else {}
                    }
                )
            
            # Connect entities to chunks
            if hasattr(graph_doc, 'source') and hasattr(graph_doc.source, 'metadata'):
                chunk_id = graph_doc.source.metadata.get('chunk_id')
                if chunk_id:
                    for node in graph_doc.nodes:
                        query = """
                        MATCH (c:Chunk {id: $chunkId})
                        MATCH (e:Entity {id: $entityId})
                        MERGE (c)-[:HAS_ENTITY]->(e)
                        """
                        
                        self.neo4j.execute_write_query(query, {
                            "chunkId": chunk_id,
                            "entityId": node.id
                        })
    
    async def _generate_chunk_embeddings(self, chunks: List[Dict[str, Any]]) -> None:
        """Generate embeddings for chunks"""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.embedding_service.generate_embeddings(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            query = """
            MATCH (c:Chunk {id: $chunkId})
            SET c.embedding = $embedding
            """
            
            self.neo4j.execute_write_query(query, {
                "chunkId": chunk["id"],
                "embedding": embedding
            })
    
    async def _update_document_status(self, document_id: str, status: ProcessingStatus) -> None:
        """Update document processing status"""
        query = """
        MATCH (d:Document {id: $docId})
        SET d.status = $status,
            d.processedAt = datetime()
        """
        
        self.neo4j.execute_write_query(query, {
            "docId": document_id,
            "status": status.value
        })
