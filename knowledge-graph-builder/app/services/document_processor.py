import logging
import asyncio
import tempfile
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import mimetypes
from datetime import datetime
import hashlib

# Document processing libraries
import PyPDF2
from docx import Document
import youtube_dl
import requests
from bs4 import BeautifulSoup
import mammoth  # For better DOCX processing

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import ServiceError
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process various document types into extractable content"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
        self.supported_types = {
            'application/pdf': self._process_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'text/plain': self._process_txt,
            'text/html': self._process_html,
            'youtube': self._process_youtube,
            'web': self._process_web
        }
    
    async def process_document(self, 
                              source: Union[str, Path], 
                              source_type: str = "auto",
                              metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a document and return structured content"""
        try:
            # Auto-detect source type if not provided
            if source_type == "auto":
                source_type = self._detect_source_type(source)
            
            # Generate unique document ID
            doc_id = self._generate_document_id(source, metadata)
            
            # Check if already processed
            if await self._is_document_processed(doc_id):
                logger.info(f"Document {doc_id} already processed, skipping")
                return await self._get_processed_document(doc_id)
            
            # Process based on type
            if source_type in self.supported_types:
                content_data = await self.supported_types[source_type](source, metadata)
            else:
                raise ServiceError(f"Unsupported document type: {source_type}")
            
            # Enhance content with metadata
            processed_doc = {
                "id": doc_id,
                "source": str(source),
                "source_type": source_type,
                "processed_at": datetime.now().isoformat(),
                "metadata": metadata or {},
                **content_data
            }
            
            # Store document metadata
            await self._store_document_metadata(processed_doc)
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Document processing failed for {source}: {e}")
            raise ServiceError(f"Document processing failed: {e}")
    
    def _detect_source_type(self, source: Union[str, Path]) -> str:
        """Auto-detect source type"""
        source_str = str(source).lower()
        
        if "youtube.com" in source_str or "youtu.be" in source_str:
            return "youtube"
        elif source_str.startswith(("http://", "https://")):
            return "web"
        elif Path(source).exists():
            mime_type, _ = mimetypes.guess_type(source)
            return mime_type or "text/plain"
        else:
            return "unknown"
    
    def _generate_document_id(self, source: Union[str, Path], metadata: Optional[Dict]) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(str(source).encode()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"doc_{timestamp}_{content_hash[:12]}"
    
    async def _is_document_processed(self, doc_id: str) -> bool:
        """Check if document is already processed"""
        query = "MATCH (d:Document {id: $docId}) RETURN count(d) > 0 as exists"
        result = self.neo4j.execute_query(query, {"docId": doc_id})
        return result[0]["exists"] if result else False
    
    async def _get_processed_document(self, doc_id: str) -> Dict[str, Any]:
        """Get already processed document"""
        query = """
        MATCH (d:Document {id: $docId})
        RETURN d.id as id, d.source as source, d.source_type as source_type,
               d.title as title, d.content as content, d.metadata as metadata
        """
        result = self.neo4j.execute_query(query, {"docId": doc_id})
        return dict(result[0]) if result else {}
    
    async def _process_pdf(self, source: Union[str, Path], metadata: Optional[Dict]) -> Dict[str, Any]:
        """Process PDF document"""
        try:
            content_parts = []
            doc_metadata = {}
            
            with open(source, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                if pdf_reader.metadata:
                    doc_metadata.update({
                        "title": pdf_reader.metadata.get('/Title', ''),
                        "author": pdf_reader.metadata.get('/Author', ''),
                        "creator": pdf_reader.metadata.get('/Creator', ''),
                        "pages": len(pdf_reader.pages)
                    })
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            content_parts.append({
                                "page": page_num,
                                "content": page_text.strip(),
                                "type": "text"
                            })
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")
            
            return {
                "title": doc_metadata.get("title", Path(source).stem),
                "content": "\n\n".join(part["content"] for part in content_parts),
                "content_parts": content_parts,
                "extracted_metadata": doc_metadata,
                "word_count": sum(len(part["content"].split()) for part in content_parts)
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise ServiceError(f"PDF processing failed: {e}")
    
    async def _process_docx(self, source: Union[str, Path], metadata: Optional[Dict]) -> Dict[str, Any]:
        """Process DOCX document"""
        try:
            # Use mammoth for better formatting preservation
            with open(source, 'rb') as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                content = result.value
                
                # Also extract with python-docx for metadata
                doc = Document(source)
                
                doc_metadata = {
                    "paragraphs": len(doc.paragraphs),
                    "title": Path(source).stem
                }
                
                # Try to extract core properties
                try:
                    core_props = doc.core_properties
                    doc_metadata.update({
                        "author": core_props.author or "",
                        "title": core_props.title or Path(source).stem,
                        "created": core_props.created.isoformat() if core_props.created else "",
                        "modified": core_props.modified.isoformat() if core_props.modified else ""
                    })
                except:
                    pass
                
                return {
                    "title": doc_metadata.get("title", Path(source).stem),
                    "content": content,
                    "extracted_metadata": doc_metadata,
                    "word_count": len(content.split())
                }
                
        except Exception as e:
            logger.error(f"DOCX processing failed: {e}")
            raise ServiceError(f"DOCX processing failed: {e}")
    
    async def _process_txt(self, source: Union[str, Path], metadata: Optional[Dict]) -> Dict[str, Any]:
        """Process plain text document"""
        try:
            with open(source, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            return {
                "title": Path(source).stem,
                "content": content,
                "extracted_metadata": {
                    "encoding": "utf-8",
                    "file_size": len(content)
                },
                "word_count": len(content.split())
            }
            
        except Exception as e:
            logger.error(f"TXT processing failed: {e}")
            raise ServiceError(f"TXT processing failed: {e}")
    
    async def _process_html(self, source: Union[str, Path], metadata: Optional[Dict]) -> Dict[str, Any]:
        """Process HTML document"""
        try:
            if str(source).startswith(('http://', 'https://')):
                # Web URL
                response = requests.get(source, timeout=30)
                response.raise_for_status()
                html_content = response.text
            else:
                # Local file
                with open(source, 'r', encoding='utf-8', errors='ignore') as file:
                    html_content = file.read()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text_content = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Extract metadata
            title = soup.title.string if soup.title else str(source)
            
            doc_metadata = {
                "title": title,
                "url": str(source) if str(source).startswith('http') else None
            }
            
            # Extract meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                if name and meta.get('content'):
                    doc_metadata[f"meta_{name}"] = meta.get('content')
            
            return {
                "title": title,
                "content": content,
                "extracted_metadata": doc_metadata,
                "word_count": len(content.split())
            }
            
        except Exception as e:
            logger.error(f"HTML processing failed: {e}")
            raise ServiceError(f"HTML processing failed: {e}")
    
    async def _process_youtube(self, url: str, metadata: Optional[Dict]) -> Dict[str, Any]:
        """Process YouTube video (extract transcript)"""
        try:
            # Configure youtube-dl options
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
            }
            
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                # Extract video info
                info = ydl.extract_info(url, download=False)
                
                title = info.get('title', 'YouTube Video')
                description = info.get('description', '')
                duration = info.get('duration', 0)
                uploader = info.get('uploader', '')
                
                # Try to get subtitles/transcript
                subtitles = info.get('subtitles', {})
                auto_subtitles = info.get('automatic_captions', {})
                
                transcript_content = ""
                
                # Prefer manual subtitles over automatic
                available_subs = subtitles or auto_subtitles
                
                if available_subs:
                    # Get English subtitles preferably
                    for lang in ['en', 'en-US', 'en-GB']:
                        if lang in available_subs:
                            # This is simplified - you'd need to download and parse the subtitle file
                            transcript_content = f"[Transcript available for {title}]"
                            break
                
                content = f"{title}\n\n{description}"
                if transcript_content:
                    content += f"\n\nTranscript:\n{transcript_content}"
                
                doc_metadata = {
                    "platform": "youtube",
                    "video_id": info.get('id', ''),
                    "duration": duration,
                    "uploader": uploader,
                    "upload_date": info.get('upload_date', ''),
                    "view_count": info.get('view_count', 0)
                }
                
                return {
                    "title": title,
                    "content": content,
                    "extracted_metadata": doc_metadata,
                    "word_count": len(content.split())
                }
                
        except Exception as e:
            logger.error(f"YouTube processing failed: {e}")
            raise ServiceError(f"YouTube processing failed: {e}")
    
    async def _process_web(self, url: str, metadata: Optional[Dict]) -> Dict[str, Any]:
        """Process web page"""
        return await self._process_html(url, metadata)
    
    async def _store_document_metadata(self, doc_data: Dict[str, Any]) -> None:
        """Store document metadata in Neo4j"""
        query = """
        MERGE (d:Document {id: $docId})
        SET d.source = $source,
            d.source_type = $sourceType,
            d.title = $title,
            d.content = $content,
            d.processed_at = datetime($processedAt),
            d.word_count = $wordCount,
            d.metadata = $metadata,
            d.status = 'processed'
        """
        
        self.neo4j.execute_write_query(query, {
            "docId": doc_data["id"],
            "source": doc_data["source"],
            "sourceType": doc_data["source_type"],
            "title": doc_data["title"],
            "content": doc_data["content"],
            "processedAt": doc_data["processed_at"],
            "wordCount": doc_data.get("word_count", 0),
            "metadata": json.dumps(doc_data.get("metadata", {}), default=str)
        })


class ChunkingService:
    """Advanced text chunking with overlap and semantic boundaries"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.sentence_endings = {'.', '!', '?', '\n\n'}
    
    def chunk_with_overlap(self, 
                          text: str, 
                          chunk_size: Optional[int] = None,
                          overlap: Optional[int] = None) -> List[Dict[str, Any]]:
        """Smart chunking that respects sentence boundaries"""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap
        
        if not text.strip():
            return []
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(self._create_chunk(current_chunk, chunk_index))
                chunk_index += 1
                
                # Start new chunk with overlap
                if overlap > 0:
                    current_chunk = self._get_overlap_text(current_chunk, overlap)
                    current_length = len(current_chunk)
                else:
                    current_chunk = ""
                    current_length = 0
            
            # Add sentence to current chunk
            if current_chunk and not current_chunk.endswith(' '):
                current_chunk += " "
                current_length += 1
            
            current_chunk += sentence
            current_length += sentence_length
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, chunk_index))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with basic sentence boundary detection"""
        import re
        
        # Simple sentence splitting - could be enhanced with NLP libraries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up and filter empty sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_chunk(self, text: str, index: int) -> Dict[str, Any]:
        """Create a chunk object with metadata"""
        return {
            "index": index,
            "text": text.strip(),
            "length": len(text),
            "word_count": len(text.split()),
            "hash": hashlib.md5(text.encode()).hexdigest()
        }
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= overlap_size:
            return text
        
        # Try to find sentence boundary within overlap region
        overlap_text = text[-overlap_size:]
        
        # Look for sentence ending within the overlap
        for ending in self.sentence_endings:
            if ending in overlap_text:
                # Start from the last sentence ending
                last_ending = overlap_text.rfind(ending)
                return overlap_text[last_ending + 1:].strip()
        
        # If no sentence boundary, just return the overlap
        return overlap_text.strip()


class BatchDocumentProcessor:
    """Process multiple documents in batches with progress tracking"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.document_processor = DocumentProcessor(neo4j_client)
        self.chunking_service = ChunkingService()
    
    async def process_batch(self, 
                           sources: List[str], 
                           batch_size: int = 5,
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Process multiple documents in batches"""
        results = {
            "total_documents": len(sources),
            "processed": 0,
            "failed": 0,
            "errors": [],
            "processing_time": 0
        }
        
        start_time = datetime.now()
        
        # Process in batches
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.document_processor.process_document(source)
                for source in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for source, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        results["failed"] += 1
                        results["errors"].append(f"{source}: {str(result)}")
                    else:
                        results["processed"] += 1
                        
                        # Process chunks for this document
                        await self._process_document_chunks(result)
                
                # Update progress
                if progress_callback:
                    progress = (i + len(batch)) / len(sources) * 100
                    await progress_callback(progress, results)
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                results["errors"].append(f"Batch error: {str(e)}")
        
        results["processing_time"] = (datetime.now() - start_time).total_seconds()
        return results
    
    async def _process_document_chunks(self, document: Dict[str, Any]) -> None:
        """Process document into chunks and store them"""
        chunks = self.chunking_service.chunk_with_overlap(document["content"])
        
        for chunk in chunks:
            await self._store_chunk(document["id"], chunk)
    
    async def _store_chunk(self, document_id: str, chunk: Dict[str, Any]) -> None:
        """Store chunk in Neo4j"""
        query = """
        MATCH (d:Document {id: $docId})
        CREATE (c:Chunk {
            id: $chunkId,
            document_id: $docId,
            index: $index,
            text: $text,
            word_count: $wordCount,
            hash: $hash
        })
        CREATE (d)-[:HAS_CHUNK]->(c)
        """
        
        chunk_id = f"{document_id}_chunk_{chunk['index']}"
        
        self.neo4j.execute_write_query(query, {
            "docId": document_id,
            "chunkId": chunk_id,
            "index": chunk["index"],
            "text": chunk["text"],
            "wordCount": chunk["word_count"],
            "hash": chunk["hash"]
        })
