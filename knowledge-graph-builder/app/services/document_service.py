import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path
import tempfile
from datetime import datetime
import uuid
from neo4j.time import DateTime as Neo4jDateTime

from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredFileLoader, YoutubeLoader,
    WikipediaLoader, WebBaseLoader, S3DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import DocumentProcessingError
from app.config.settings import get_settings
from app.models.responses import DocumentInfo, ProcessingStatus, ProcessingProgress
from app.models.requests import DocumentSource

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
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
            
            self.neo4j.execute_write_query(query, {
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
    
    async def _scan_s3_bucket(self, bucket_name: str, prefix: str = "") -> List[DocumentInfo]:
        """Scan S3 bucket for documents"""
        documents = []
        try:
            loader = S3DirectoryLoader(bucket=bucket_name, prefix=prefix)
            # This would need proper implementation with boto3
            # For now, returning placeholder
            logger.info(f"Scanning S3 bucket: {bucket_name}")
            return documents
        except Exception as e:
            raise DocumentProcessingError(f"Failed to scan S3 bucket: {e}")
    
    async def _scan_gcs_bucket(self, project_id: str, bucket_name: str, prefix: str = "") -> List[DocumentInfo]:
        """Scan GCS bucket for documents"""
        documents = []
        try:
            # GCS implementation would go here
            logger.info(f"Scanning GCS bucket: {bucket_name}")
            return documents
        except Exception as e:
            raise DocumentProcessingError(f"Failed to scan GCS bucket: {e}")
    
    async def _scan_youtube_video(self, video_url: str) -> List[DocumentInfo]:
        """Scan YouTube video for transcript"""
        try:
            doc_id = str(uuid.uuid4())
            
            # Extract video ID from URL
            video_id = video_url.split('v=')[-1].split('&')[0]
            
            query = """
            CREATE (d:Document {
                id: $id,
                fileName: $fileName,
                url: $url,
                sourceType: $sourceType,
                status: $status,
                createdAt: datetime()
            })
            RETURN d
            """
            
            self.neo4j.execute_write_query(query, {
                "id": doc_id,
                "fileName": f"youtube_video_{video_id}",
                "url": video_url,
                "sourceType": DocumentSource.YOUTUBE.value,
                "status": ProcessingStatus.NEW.value
            })
            
            return [DocumentInfo(
                id=doc_id,
                file_name=f"youtube_video_{video_id}",
                source_type=DocumentSource.YOUTUBE.value,
                status=ProcessingStatus.NEW,
                created_at=datetime.now()
            )]
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to scan YouTube video: {e}")
    
    async def _scan_wikipedia_page(self, page_title: str) -> List[DocumentInfo]:
        """Scan Wikipedia page"""
        try:
            doc_id = str(uuid.uuid4())
            
            query = """
            CREATE (d:Document {
                id: $id,
                fileName: $fileName,
                pageTitle: $pageTitle,
                sourceType: $sourceType,
                status: $status,
                createdAt: datetime()
            })
            RETURN d
            """
            
            self.neo4j.execute_write_query(query, {
                "id": doc_id,
                "fileName": f"wikipedia_{page_title.replace(' ', '_')}",
                "pageTitle": page_title,
                "sourceType": DocumentSource.WIKIPEDIA.value,
                "status": ProcessingStatus.NEW.value
            })
            
            return [DocumentInfo(
                id=doc_id,
                file_name=f"wikipedia_{page_title.replace(' ', '_')}",
                source_type=DocumentSource.WIKIPEDIA.value,
                status=ProcessingStatus.NEW,
                created_at=datetime.now()
            )]
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to scan Wikipedia page: {e}")
    
    async def _scan_web_page(self, url: str) -> List[DocumentInfo]:
        """Scan web page"""
        try:
            doc_id = str(uuid.uuid4())
            
            query = """
            CREATE (d:Document {
                id: $id,
                fileName: $fileName,
                url: $url,
                sourceType: $sourceType,
                status: $status,
                createdAt: datetime()
            })
            RETURN d
            """
            
            self.neo4j.execute_write_query(query, {
                "id": doc_id,
                "fileName": f"web_page_{hash(url)}",
                "url": url,
                "sourceType": DocumentSource.WEB.value,
                "status": ProcessingStatus.NEW.value
            })
            
            return [DocumentInfo(
                id=doc_id,
                file_name=f"web_page_{hash(url)}",
                source_type=DocumentSource.WEB.value,
                status=ProcessingStatus.NEW,
                created_at=datetime.now()
            )]
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to scan web page: {e}")
    
    async def load_document_content(self, document_info: DocumentInfo) -> List[Document]:
        """Load actual document content using appropriate loader"""
        try:
            if document_info.source_type == DocumentSource.LOCAL.value:
                return await self._load_local_file(document_info)
            elif document_info.source_type == DocumentSource.YOUTUBE.value:
                return await self._load_youtube_content(document_info)
            elif document_info.source_type == DocumentSource.WIKIPEDIA.value:
                return await self._load_wikipedia_content(document_info)
            elif document_info.source_type == DocumentSource.WEB.value:
                return await self._load_web_content(document_info)
            else:
                raise DocumentProcessingError(f"Unsupported source type: {document_info.source_type}")
                
        except Exception as e:
            logger.error(f"Error loading document {document_info.file_name}: {e}")
            raise DocumentProcessingError(f"Failed to load document content: {e}")
    
    async def _load_local_file(self, document_info: DocumentInfo) -> List[Document]:
        """Load local file content"""
        # Get file path from Neo4j
        query = "MATCH (d:Document {id: $id}) RETURN d.filePath as filePath"
        result = self.neo4j.execute_query(query, {"id": document_info.id})
        
        if not result:
            raise DocumentProcessingError("Document not found")
        
        file_path = result[0]["filePath"]
        
        if file_path.endswith('.pdf'):
            loader = PyMuPDFLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
        
        documents = await asyncio.to_thread(loader.load)
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "document_id": document_info.id,
                "file_name": document_info.file_name,
                "source_type": document_info.source_type
            })
        
        return documents
    
    async def _load_youtube_content(self, document_info: DocumentInfo) -> List[Document]:
        """Load YouTube video transcript"""
        query = "MATCH (d:Document {id: $id}) RETURN d.url as url"
        result = self.neo4j.execute_query(query, {"id": document_info.id})
        
        if not result:
            raise DocumentProcessingError("Document not found")
        
        video_url = result[0]["url"]
        
        loader = YoutubeLoader.from_youtube_url(
            video_url, 
            add_video_info=True,
            language=["en", "en-US"]
        )
        
        documents = await asyncio.to_thread(loader.load)
        
        for doc in documents:
            doc.metadata.update({
                "document_id": document_info.id,
                "file_name": document_info.file_name,
                "source_type": document_info.source_type,
                "url": video_url
            })
        
        return documents
    
    async def _load_wikipedia_content(self, document_info: DocumentInfo) -> List[Document]:
        """Load Wikipedia page content"""
        query = "MATCH (d:Document {id: $id}) RETURN d.pageTitle as pageTitle"
        result = self.neo4j.execute_query(query, {"id": document_info.id})
        
        if not result:
            raise DocumentProcessingError("Document not found")
        
        page_title = result[0]["pageTitle"]
        
        loader = WikipediaLoader(
            query=page_title,
            load_max_docs=1
        )
        
        documents = await asyncio.to_thread(loader.load)
        
        for doc in documents:
            doc.metadata.update({
                "document_id": document_info.id,
                "file_name": document_info.file_name,
                "source_type": document_info.source_type,
                "page_title": page_title
            })
        
        return documents
    
    async def _load_web_content(self, document_info: DocumentInfo) -> List[Document]:
        """Load web page content"""
        query = "MATCH (d:Document {id: $id}) RETURN d.url as url"
        result = self.neo4j.execute_query(query, {"id": document_info.id})
        
        if not result:
            raise DocumentProcessingError("Document not found")
        
        url = result[0]["url"]
        
        loader = WebBaseLoader([url])
        documents = await asyncio.to_thread(loader.load)
        
        for doc in documents:
            doc.metadata.update({
                "document_id": document_info.id,
                "file_name": document_info.file_name,
                "source_type": document_info.source_type,
                "url": url
            })
        
        return documents
    
    async def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            chunks = await asyncio.to_thread(self.text_splitter.split_documents, documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise DocumentProcessingError(f"Failed to split documents: {e}")
    
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
            
            self.neo4j.execute_write_query(update_doc_query, {
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
                    chunk_query += "CREATE (d)-[:FIRST_CHUNK]->(c)"
                
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
                
                self.neo4j.execute_write_query(chunk_query, params)
            
            logger.info(f"Created {len(chunks)} chunk nodes for document {document_info.file_name}")
            
        except Exception as e:
            logger.error(f"Error creating chunk nodes: {e}")
            raise DocumentProcessingError(f"Failed to create chunk nodes: {e}")
    
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
        
        result = self.neo4j.execute_query(query)
        
        documents = []
        for record in result:
            created_at = record["createdAt"]
            processed_at = record.get("processedAt")
            # Convert Neo4jDateTime to Python datetime
            if isinstance(created_at, Neo4jDateTime):
                created_at = created_at.to_native()
            if isinstance(processed_at, Neo4jDateTime):
                processed_at = processed_at.to_native()
            documents.append(DocumentInfo(
                id=record["id"],
                file_name=record["fileName"],
                source_type=record["sourceType"],
                status=ProcessingStatus(record["status"]),
                created_at=created_at,  # <-- use converted variable
                processed_at=processed_at,  # <-- use converted variable
                chunk_count=record.get("chunkCount", 0)
            ))
        
        return documents
