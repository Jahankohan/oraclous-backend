"""
Document Service - Async version using Neo4j Pool
This is an example of how to migrate services to use the new async pool
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path
import tempfile
from datetime import datetime
import uuid

from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredFileLoader, YoutubeLoader,
    WikipediaLoader, WebBaseLoader, S3DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.core.neo4j_pool import Neo4jPool
from app.core.exceptions import DocumentProcessingError
from app.config.settings import get_settings
from app.models.responses import DocumentInfo, ProcessingStatus, ProcessingProgress
from app.models.requests import DocumentSource

logger = logging.getLogger(__name__)


class DocumentServiceAsync:
    """Async version of DocumentService using Neo4j Pool"""
    
    def __init__(self, neo4j_pool: Neo4jPool):
        self.neo4j = neo4j_pool
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    async def scan_sources(self, source_type: DocumentSource, **kwargs) -> List[DocumentInfo]:
        """Scan and create document source nodes"""
        try:
            if source_type == DocumentSource.LOCAL:
                return await self._scan_local_files(**kwargs)
            elif source_type == DocumentSource.S3:
                return await self._scan_s3_bucket(**kwargs)
            elif source_type == DocumentSource.GCS:
                return await self._scan_gcs_bucket(**kwargs)
            elif source_type == DocumentSource.YOUTUBE:
                return await self._scan_youtube_video(**kwargs)
            elif source_type == DocumentSource.WIKIPEDIA:
                return await self._scan_wikipedia_page(**kwargs)
            elif source_type == DocumentSource.WEB:
                return await self._scan_web_page(**kwargs)
            else:
                raise DocumentProcessingError(f"Unsupported source type: {source_type}")
                
        except Exception as e:
            logger.error(f"Error scanning {source_type} sources: {e}")
            raise DocumentProcessingError(f"Failed to scan sources: {e}")
    
    async def _scan_local_files(self, file_paths: List[str]) -> List[DocumentInfo]:
        """Scan local files"""
        documents = []
        for file_path in file_paths:
            doc_id = str(uuid.uuid4())
            file_name = Path(file_path).name
            
            # Create document node in Neo4j
            query = """
            CREATE (d:Document {
                id: $id,
                fileName: $fileName,
                filePath: $filePath,
                sourceType: $sourceType,
                status: $status,
                createdAt: datetime()
            })
            RETURN d
            """
            
            await self.neo4j.execute_write(query, {
                "id": doc_id,
                "fileName": file_name,
                "filePath": file_path,
                "sourceType": DocumentSource.LOCAL.value,
                "status": ProcessingStatus.NEW.value
            })
            
            documents.append(DocumentInfo(
                id=doc_id,
                file_name=file_name,
                source_type=DocumentSource.LOCAL.value,
                status=ProcessingStatus.NEW,
                created_at=datetime.now()
            ))
        
        return documents
    
    async def get_documents_list(self) -> List[DocumentInfo]:
        """Get list of all documents"""
        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
        RETURN d.id as id,
               d.fileName as fileName,
               d.sourceType as sourceType,
               d.status as status,
               d.createdAt as createdAt,
               d.processedAt as processedAt,
               count(c) as chunkCount
        ORDER BY d.createdAt DESC
        """
        
        result = await self.neo4j.execute_read(query)
        
        documents = []
        for record in result:
            documents.append(DocumentInfo(
                id=record["id"],
                file_name=record["fileName"],
                source_type=record["sourceType"],
                status=ProcessingStatus(record["status"]),
                created_at=record["createdAt"],
                processed_at=record.get("processedAt"),
                chunk_count=record.get("chunkCount", 0)
            ))
        
        return documents
    
    async def create_chunk_nodes(self, document_info: DocumentInfo, chunks: List[Document]) -> None:
        """Create chunk nodes and relationships in Neo4j"""
        try:
            # Update document status to processing
            update_doc_query = """
            MATCH (d:Document {id: $doc_id})
            SET d.status = $status,
                d.totalChunks = $total_chunks,
                d.processedAt = datetime()
            """
            
            await self.neo4j.execute_write(update_doc_query, {
                "doc_id": document_info.id,
                "status": ProcessingStatus.PROCESSING.value,
                "total_chunks": len(chunks)
            })
            
            # Create chunk nodes and relationships
            for i, chunk in enumerate(chunks):
                chunk_query = """
                MATCH (d:Document {id: $doc_id})
                CREATE (c:Chunk {
                    id: $chunk_id,
                    text: $text,
                    chunkIndex: $chunk_index,
                    createdAt: datetime()
                })
                CREATE (d)-[:HAS_CHUNK]->(c)
                """
                
                if i == 0:
                    chunk_query += " CREATE (d)-[:FIRST_CHUNK]->(c)"
                
                if i > 0:
                    chunk_query += """
                    WITH c
                    MATCH (prev:Chunk {id: $prev_chunk_id})
                    CREATE (prev)-[:NEXT_CHUNK]->(c)
                    """
                
                params = {
                    "doc_id": document_info.id,
                    "chunk_id": chunk.metadata["chunk_id"],
                    "text": chunk.page_content,
                    "chunk_index": i
                }
                
                if i > 0:
                    params["prev_chunk_id"] = chunks[i-1].metadata["chunk_id"]
                
                await self.neo4j.execute_write(chunk_query, params)
            
            logger.info(f"Created {len(chunks)} chunk nodes for document {document_info.file_name}")
            
        except Exception as e:
            logger.error(f"Error creating chunk nodes: {e}")
            raise DocumentProcessingError(f"Failed to create chunk nodes: {e}")
    
    # Additional methods would be migrated similarly...