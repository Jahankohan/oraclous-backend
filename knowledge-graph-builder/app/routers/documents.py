# File: app/routers/documents.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional
import json
import asyncio
from datetime import datetime

from app.core.neo4j_client import Neo4jClient, get_neo4j_client
from app.models.requests import DocumentUploadRequest, ExtractionRequest, ProcessingMode
from app.models.responses import BaseResponse, DocumentInfo, ProcessingProgress
from app.services.advanced_graph_integration_service import AdvancedGraphIntegrationService
from app.services.advanced_graph_integration_service import AdvancedGraphIntegrationService
from app.services.extraction_service import ExtractionService
from app.services.document_service import DocumentService

router = APIRouter()

@router.post("/upload", response_model=BaseResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Upload files for processing"""
    try:
        integration_service = AdvancedGraphIntegrationService(neo4j)
        uploaded_files = []
        for file in files:
            # Save file temporarily
            import tempfile
            import shutil
            from pathlib import Path
            temp_dir = Path(tempfile.gettempdir()) / "llm_graph_builder"
            temp_dir.mkdir(exist_ok=True)
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(str(file_path))
        documents = await integration_service.scan_sources("local", file_paths=uploaded_files)
        return BaseResponse(
            success=True,
            message=f"Successfully uploaded {len(documents)} files",
            data={"uploaded_files": [doc["file_name"] for doc in documents]}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/url/scan", response_model=BaseResponse)
async def scan_url_sources(
    request: DocumentUploadRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Scan and create document source nodes from URLs"""
    try:
        document_service = DocumentService(neo4j)
        
        documents = []
        if request.source_type.value == "youtube" and request.url:
            documents = await document_service.scan_sources("youtube", video_url=str(request.url))
        elif request.source_type.value == "wiki" and request.content:
            documents = await document_service.scan_sources("wiki", page_title=request.content)
        elif request.source_type.value == "web" and request.url:
            documents = await document_service.scan_sources("web", url=str(request.url))
        elif request.source_type.value == "s3" and request.s3_bucket:
            documents = await document_service.scan_sources("s3", bucket_name=request.s3_bucket, prefix=request.s3_key or "")
        elif request.source_type.value == "gcs" and request.gcs_bucket:
            documents = await document_service.scan_sources("gcs", 
                project_id=request.gcs_project_id, 
                bucket_name=request.gcs_bucket, 
                prefix=request.gcs_blob_name or "")
        else:
            raise HTTPException(status_code=400, detail="Invalid source configuration")
        
        return BaseResponse(
            success=True,
            message=f"Successfully scanned {len(documents)} sources",
            data={"sources": [{"id": doc.id, "name": doc.file_name} for doc in documents]}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Source scan failed: {str(e)}")

@router.post("/extract")
async def extract_graph_from_documents(
    request: ExtractionRequest,
    background_tasks: BackgroundTasks,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Extract knowledge graph from documents with real-time progress updates"""
    try:
        extraction_service = ExtractionService(neo4j)
        from app.services.document_service import DocumentService
        document_service = DocumentService(neo4j)

        async def generate_progress():
            async for progress in extraction_service.extract_graph(
                file_names=request.file_names,
                model=request.model,
                node_labels=request.node_labels,
                relationship_types=request.relationship_types,
                enable_schema=request.enable_schema,
                document_service=document_service
            ):
                yield f"data: {json.dumps(progress.dict())}\n\n"

        return StreamingResponse(
            generate_progress(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@router.get("/sources_list", response_model=List[DocumentInfo])
async def get_sources_list(
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> List[DocumentInfo]:
    """Get list of all document sources"""
    try:
        integration_service = AdvancedGraphIntegrationService(neo4j)
        docs = await integration_service.get_documents_list()
        # Convert dicts to DocumentInfo if needed
        doc_objs = []
        for doc in docs:
            if isinstance(doc, dict):
                doc_obj = DocumentInfo(
                    id=doc.get("id", ""),
                    file_name=doc.get("file_name") or doc.get("fileName") or "Unknown",
                    source_type=doc.get("source_type") or doc.get("sourceType") or "local",
                    status=doc.get("status") or "New",
                    created_at=doc.get("created_at") or doc.get("createdAt") or datetime.now(),
                    processed_at=doc.get("processed_at") or doc.get("processedAt"),
                    node_count=doc.get("node_count"),
                    relationship_count=doc.get("relationship_count"),
                    chunk_count=doc.get("chunk_count")
                )
                doc_objs.append(doc_obj)
            else:
                # Already a DocumentInfo object
                doc_objs.append(doc)
        return doc_objs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sources list: {str(e)}")

@router.get("/update_extract_status/{file_name}")
async def get_extraction_status(
    file_name: str,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Get real-time extraction status for a file"""
    try:
        async def generate_status():
            # Get document info
            query = """
            MATCH (d:Document {fileName: $fileName})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            RETURN d.status as status, 
                   coalesce(d.totalChunks, 0) as totalChunks,
                   count(c) as processedChunks
            """
            
            while True:
                result = neo4j.execute_query(query, {"fileName": file_name})
                
                if result:
                    record = result[0]
                    progress = ProcessingProgress(
                        file_name=file_name,
                        status=record["status"],
                        progress_percentage=(record["processedChunks"] / max(record["totalChunks"], 1)) * 100,
                        chunks_processed=record["processedChunks"],
                        total_chunks=record["totalChunks"],
                        current_step="Processing"
                    )
                    
                    yield f"data: {json.dumps(progress.dict())}\n\n"
                    
                    # Stop if completed or failed
                    if record["status"] in ["Completed", "Failed", "Cancelled"]:
                        break
                
                await asyncio.sleep(1)
        
        return StreamingResponse(
            generate_status(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/canceled_job", response_model=BaseResponse)
async def cancel_processing_job(
    file_name: str,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Cancel a processing job"""
    try:
        query = """
        MATCH (d:Document {fileName: $fileName})
        WHERE d.status = 'Processing'
        SET d.status = 'Cancelled'
        RETURN d.fileName as fileName
        """
        
        result = neo4j.execute_write_query(query, {"fileName": file_name})
        
        if result:
            return BaseResponse(
                success=True,
                message=f"Job cancelled for {file_name}"
            )
        else:
            raise HTTPException(status_code=404, detail="Job not found or not in processing state")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@router.post("/retry_processing", response_model=BaseResponse)
async def retry_processing(
    file_name: str,
    mode: ProcessingMode,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Retry processing a failed or cancelled document"""
    try:
        if mode == ProcessingMode.DELETE_AND_RESTART:
            # Delete existing entities and chunks
            query = """
            MATCH (d:Document {fileName: $fileName})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:Entity)
            DETACH DELETE c, e
            SET d.status = 'New'
            RETURN d.fileName as fileName
            """
        else:
            # Just reset status
            query = """
            MATCH (d:Document {fileName: $fileName})
            SET d.status = 'New'
            RETURN d.fileName as fileName
            """
        
        result = neo4j.execute_write_query(query, {"fileName": file_name})
        
        if result:
            return BaseResponse(
                success=True,
                message=f"Document {file_name} ready for reprocessing"
            )
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retry processing: {str(e)}")

@router.post("/post_processing", response_model=BaseResponse)
async def run_post_processing(
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Run post-processing tasks on the graph"""
    try:
        from app.services.graph_service import GraphService
        graph_service = GraphService(neo4j)
        
        results = await graph_service.post_process_graph()
        
        return BaseResponse(
            success=True,
            message="Post-processing completed successfully",
            data=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Post-processing failed: {str(e)}")

@router.get("/chunk_entities")
async def get_chunk_entities(
    chunk_id: str,
    mode: str = "vector",
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Get entities and relationships for a specific chunk"""
    try:
        if mode == "vector":
            query = """
            MATCH (c:Chunk {id: $chunkId})-[:HAS_ENTITY]->(e:Entity)
            OPTIONAL MATCH (e)-[r]-(related:Entity)
            RETURN c.text as chunkText,
                   collect(DISTINCT {
                       id: e.id,
                       name: coalesce(e.name, e.id),
                       labels: labels(e),
                       properties: properties(e)
                   }) as entities,
                   collect(DISTINCT {
                       source: e.id,
                       target: related.id,
                       type: type(r),
                       properties: properties(r)
                   }) as relationships
            """
        else:
            # Add full-text or other modes here
            query = """
            MATCH (c:Chunk {id: $chunkId})
            RETURN c.text as chunkText, [] as entities, [] as relationships
            """
        
        result = neo4j.execute_query(query, {"chunkId": chunk_id})
        
        if result:
            return result[0]
        else:
            raise HTTPException(status_code=404, detail="Chunk not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chunk entities: {str(e)}")

@router.get("/fetch_chunktext")
async def fetch_chunk_text(
    chunk_id: str,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Fetch text and metadata for a specific chunk"""
    try:
        query = """
        MATCH (c:Chunk {id: $chunkId})
        MATCH (d:Document)-[:HAS_CHUNK]->(c)
        RETURN c.text as text,
               c.chunkIndex as chunkIndex,
               d.fileName as fileName,
               d.sourceType as sourceType,
               properties(c) as metadata
        """
        
        result = neo4j.execute_query(query, {"chunkId": chunk_id})
        
        if result:
            return result[0]
        else:
            raise HTTPException(status_code=404, detail="Chunk not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chunk text: {str(e)}")
