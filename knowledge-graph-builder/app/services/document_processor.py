"""
Document Processing Service

Handles different file types for ingestion:
- Raw text
- PDF files  
- DOC/DOCX files
- Future: URLs, structured data

This service prepares documents for the GraphRAG pipeline.
"""
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status

from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Process different document types for GraphRAG ingestion.
    
    Supported formats:
    - text: Raw text content
    - pdf: PDF files (future implementation)
    - doc/docx: Word documents (future implementation)
    - url: Web content (future implementation)
    """
    
    @staticmethod
    def process_document(
        content: str, 
        source_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process document content based on source type.
        
        Args:
            content: Document content (text or base64 for files)
            source_type: Type of document ("text", "pdf", "doc", "docx", "url")
            metadata: Additional metadata for the document
            
        Returns:
            Processed document with text content and metadata
        """
        
        if source_type == "text":
            return DocumentProcessor._process_text(content, metadata)
        elif source_type == "pdf":
            return DocumentProcessor._process_pdf(content, metadata)
        elif source_type in ["doc", "docx"]:
            return DocumentProcessor._process_word(content, metadata)
        elif source_type == "url":
            return DocumentProcessor._process_url(content, metadata)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported source type: {source_type}. Supported types: text, pdf, doc, docx, url"
            )
    
    @staticmethod
    def _process_text(content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process raw text content."""
        if not content or len(content.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text content must be at least 10 characters long"
            )
        
        processed_metadata = metadata or {}
        processed_metadata.update({
            "content_length": len(content),
            "content_type": "text/plain",
            "processing_method": "raw_text"
        })
        
        return {
            "text": content.strip(),
            "metadata": processed_metadata,
            "success": True
        }
    
    @staticmethod
    def _process_pdf(content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a PDF file.

        Args:
            content: Absolute path to the PDF file on disk (stored in source_content).
            metadata: Additional metadata dict.
        """
        from app.services.pdf_extractor import extract_pdf

        try:
            result = extract_pdf(content)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"PDF extraction failed: {exc}",
            )

        if not result["text"].strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from this PDF.",
            )

        processed_metadata = metadata or {}
        processed_metadata.update(result["metadata"])
        processed_metadata["content_length"] = len(result["text"])

        return {
            "text": result["text"],
            "metadata": processed_metadata,
            "image_paths": result.get("image_paths", []),
            "success": True,
        }

    @staticmethod
    def _process_word(content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a DOCX file.

        Args:
            content: Absolute path to the DOCX file on disk.
            metadata: Additional metadata dict.
        """
        from app.services.pdf_extractor import extract_docx

        try:
            result = extract_docx(content)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"DOCX extraction failed: {exc}",
            )

        if not result["text"].strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from this DOCX file.",
            )

        processed_metadata = metadata or {}
        processed_metadata.update(result["metadata"])
        processed_metadata["content_length"] = len(result["text"])

        return {
            "text": result["text"],
            "metadata": processed_metadata,
            "image_paths": [],
            "success": True,
        }
    
    @staticmethod
    def _process_url(content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process URL content by fetching and extracting text.
        
        TODO: Implement web scraping using:
        - requests + BeautifulSoup
        - newspaper3k for articles
        - readability-lxml for clean text extraction
        """
        logger.warning("URL processing not yet implemented")
        
        # For now, return error
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="URL processing is not yet implemented. Coming soon!"
        )
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """Return list of supported document types."""
        return ["text", "pdf", "doc", "docx"]
    
    @staticmethod
    def validate_source_type(source_type: str) -> bool:
        """Validate if source type is supported."""
        return source_type in DocumentProcessor.get_supported_types()


# Global instance
document_processor = DocumentProcessor()
