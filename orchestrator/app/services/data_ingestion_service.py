# orchestrator/app/services/data_ingestion_service.py
import asyncio
import hashlib
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from app.repositories.data_source_repository import DataSourceRepository
from app.repositories.ingestion_job_repository import IngestionJobRepository
from app.repositories.document_repository import DocumentRepository
from app.services.data_ingestion import IngestionRegistry, BaseIngestor
from app.services.data_ingestion.base import DataSource as BaseDataSource, IngestionJob as BaseIngestionJob
from app.schemas.data_source_schema import DataSourceCreate, DataSourceUpdate, DataSourceRead
from app.schemas.ingestion_job_schema import IngestionJobCreate, IngestionJobUpdate, IngestionJobRead
from app.schemas.document_schema import DocumentCreate
from app.models.ingestion_job_model import IngestionStatus
from app.models.data_source_model import DataSource
from app.models.ingestion_job_model import IngestionJob
import logging

logger = logging.getLogger(__name__)

class DataIngestionService:
    def __init__(
        self,
        data_source_repo: DataSourceRepository,
        ingestion_job_repo: IngestionJobRepository,
        document_repo: DocumentRepository,
        ingestion_registry: IngestionRegistry
    ):
        self.data_source_repo = data_source_repo
        self.ingestion_job_repo = ingestion_job_repo
        self.document_repo = document_repo
        self.registry = ingestion_registry
        self._running_jobs: Dict[UUID, asyncio.Task] = {}

    # Data Source Management
    async def create_data_source(self, data: DataSourceCreate) -> DataSourceRead:
        """Create a new data source"""
        # Validate that we have an ingestor for this source type
        ingestor = self.registry.get_ingestor(data.type.value)
        if not ingestor:
            raise ValueError(f"No ingestor available for source type: {data.type}")
        
        # Validate the configuration
        if not await ingestor.validate_config(data.config):
            raise ValueError(f"Invalid configuration for source type: {data.type}")
        
        source = await self.data_source_repo.create(data)
        return DataSourceRead.model_validate(source)

    async def get_data_source(self, source_id: UUID) -> Optional[DataSourceRead]:
        """Get a data source by ID"""
        source = await self.data_source_repo.get_by_id(source_id)
        if source:
            return DataSourceRead.model_validate(source)
        return None

    async def update_data_source(self, source_id: UUID, data: DataSourceUpdate) -> Optional[DataSourceRead]:
        """Update a data source"""
        source = await self.data_source_repo.update(source_id, data)
        if source:
            return DataSourceRead.model_validate(source)
        return None

    async def delete_data_source(self, source_id: UUID) -> bool:
        """Delete a data source and all related jobs/documents"""
        # Cancel any running jobs for this source
        jobs = await self.ingestion_job_repo.get_by_source(source_id)
        for job in jobs:
            if job.status == IngestionStatus.RUNNING:
                await self.cancel_job(job.id)
        
        return await self.data_source_repo.delete(source_id)

    async def list_data_sources(self, owner_id: Optional[str] = None) -> List[DataSourceRead]:
        """List data sources, optionally filtered by owner"""
        if owner_id:
            sources = await self.data_source_repo.get_by_owner(owner_id)
        else:
            sources = await self.data_source_repo.list_all()
        
        return [DataSourceRead.model_validate(source) for source in sources]

    async def test_data_source(self, source_id: UUID) -> bool:
        """Test connection to a data source"""
        source = await self.data_source_repo.get_by_id(source_id)
        if not source:
            raise ValueError("Data source not found")
        
        ingestor = self.registry.get_ingestor(source.type.value)
        if not ingestor:
            raise ValueError(f"No ingestor available for source type: {source.type}")
        
        # Convert to base data source format
        base_source = BaseDataSource(
            id=str(source.id),
            name=source.name,
            type=source.type.value,
            config=source.config,
            credentials={"user_id": source.credentials_ref} if source.credentials_ref else None
        )
        
        return await ingestor.test_connection(base_source)

    async def get_available_resources(self, source_id: UUID) -> List[Dict[str, Any]]:
        """Get available resources from a data source"""
        source = await self.data_source_repo.get_by_id(source_id)
        if not source:
            raise ValueError("Data source not found")
        
        ingestor = self.registry.get_ingestor(source.type.value)
        if not ingestor:
            raise ValueError(f"No ingestor available for source type: {source.type}")
        
        base_source = BaseDataSource(
            id=str(source.id),
            name=source.name,
            type=source.type.value,
            config=source.config,
            credentials={"user_id": source.credentials_ref} if source.credentials_ref else None
        )
        
        return await ingestor.get_available_resources(base_source)

    # Ingestion Job Management
    async def create_ingestion_job(self, data: IngestionJobCreate) -> IngestionJobRead:
        """Create a new ingestion job"""
        # Verify the data source exists
        source = await self.data_source_repo.get_by_id(data.source_id)
        if not source:
            raise ValueError("Data source not found")
        
        job = await self.ingestion_job_repo.create(data)
        return IngestionJobRead.model_validate(job)

    async def start_ingestion_job(self, job_id: UUID, config_override: Optional[Dict[str, Any]] = None) -> bool:
        """Start an ingestion job"""
        job = await self.ingestion_job_repo.get_by_id(job_id)
        if not job:
            raise ValueError("Ingestion job not found")
        
        if job.status != IngestionStatus.PENDING:
            raise ValueError(f"Job is not in pending status: {job.status}")
        
        # Check if job is already running
        if job_id in self._running_jobs:
            raise ValueError("Job is already running")
        
        # Start the job as a background task
        task = asyncio.create_task(self._execute_ingestion_job(job_id, config_override))
        self._running_jobs[job_id] = task
        
        # Update job status to running
        await self.ingestion_job_repo.update(job_id, IngestionJobUpdate(
            status=IngestionStatus.RUNNING,
            started_at=datetime.utcnow()
        ))
        
        return True

    async def cancel_job(self, job_id: UUID) -> bool:
        """Cancel a running ingestion job"""
        if job_id in self._running_jobs:
            task = self._running_jobs[job_id]
            task.cancel()
            del self._running_jobs[job_id]
            
            await self.ingestion_job_repo.update(job_id, IngestionJobUpdate(
                status=IngestionStatus.CANCELLED,
                completed_at=datetime.utcnow()
            ))
            return True
        
        return False

    async def get_ingestion_job(self, job_id: UUID) -> Optional[IngestionJobRead]:
        """Get an ingestion job by ID"""
        job = await self.ingestion_job_repo.get_by_id(job_id)
        if job:
            return IngestionJobRead.model_validate(job)
        return None

    async def list_ingestion_jobs(self, source_id: Optional[UUID] = None, status: Optional[IngestionStatus] = None) -> List[IngestionJobRead]:
        """List ingestion jobs with optional filters"""
        if source_id:
            jobs = await self.ingestion_job_repo.get_by_source(source_id)
        elif status:
            jobs = await self.ingestion_job_repo.get_by_status(status)
        else:
            jobs = await self.ingestion_job_repo.list_all()
        
        return [IngestionJobRead.model_validate(job) for job in jobs]

    async def _execute_ingestion_job(self, job_id: UUID, config_override: Optional[Dict[str, Any]] = None):
        """Execute an ingestion job in the background"""
        try:
            # Get job and source details
            job = await self.ingestion_job_repo.get_by_id(job_id)
            source = await self.data_source_repo.get_by_id(job.source_id)
            
            if not job or not source:
                raise ValueError("Job or source not found")
            
            # Get the appropriate ingestor
            ingestor = self.registry.get_ingestor(source.type.value)
            if not ingestor:
                raise ValueError(f"No ingestor available for source type: {source.type}")
            
            # Prepare the ingestion parameters
            base_source = BaseDataSource(
                id=str(source.id),
                name=source.name,
                type=source.type.value,
                config=source.config,
                credentials={"user_id": source.credentials_ref} if source.credentials_ref else None
            )
            
            # Merge job config with override
            job_config = job.config or {}
            if config_override:
                job_config.update(config_override)
            
            base_job = BaseIngestionJob(
                id=str(job.id),
                source_id=str(source.id),
                config=job_config,
                user_id=job.owner_id
            )
            
            # Execute the ingestion
            result = await ingestor.ingest(base_source, base_job)
            
            # Save the documents
            documents_created = 0
            if result.documents:
                doc_creates = []
                for doc_data in result.documents:
                    content_hash = hashlib.sha256(doc_data["text"].encode()).hexdigest()
                    
                    # Check for duplicates
                    existing = await self.document_repo.get_by_hash(content_hash)
                    if existing:
                        logger.info(f"Skipping duplicate document: {content_hash}")
                        continue
                    
                    doc_create = DocumentCreate(
                        job_id=job.id,
                        source_id=source.id,
                        external_id=doc_data.get("id"),
                        title=doc_data.get("metadata", {}).get("title"),
                        content=doc_data["text"],
                        content_type="text",
                        metadata=doc_data.get("metadata", {}),
                        content_hash=content_hash,
                        size_bytes=len(doc_data["text"].encode()),
                        owner_id=job.owner_id
                    )
                    doc_creates.append(doc_create)
                
                if doc_creates:
                    await self.document_repo.create_many(doc_creates)
                    documents_created = len(doc_creates)
            
            # Update job status
            await self.ingestion_job_repo.update(job_id, IngestionJobUpdate(
                status=IngestionStatus.COMPLETED,
                documents_count=documents_created,
                metadata=result.metadata,
                completed_at=datetime.utcnow()
            ))
            
            logger.info(f"Ingestion job {job_id} completed successfully. {documents_created} documents created.")
            
        except asyncio.CancelledError:
            logger.info(f"Ingestion job {job_id} was cancelled")
            await self.ingestion_job_repo.update(job_id, IngestionJobUpdate(
                status=IngestionStatus.CANCELLED,
                completed_at=datetime.utcnow()
            ))
        except Exception as e:
            logger.error(f"Ingestion job {job_id} failed: {e}")
            await self.ingestion_job_repo.update(job_id, IngestionJobUpdate(
                status=IngestionStatus.FAILED,
                error_message=str(e),
                completed_at=datetime.utcnow()
            ))
        finally:
            # Clean up running job tracking
            if job_id in self._running_jobs:
                del self._running_jobs[job_id]

    # Document Management
    async def search_documents(self, query: str, limit: int = 50, source_ids: Optional[List[UUID]] = None) -> List[Dict[str, Any]]:
        """Search documents by content"""
        # For now, simple text search. In production, you'd want proper vector search
        documents = await self.document_repo.search_by_content(query, limit)
        
        # Filter by source_ids if provided
        if source_ids:
            documents = [doc for doc in documents if doc.source_id in source_ids]
        
        return [
            {
                "id": str(doc.id),
                "title": doc.title,
                "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                "source_id": str(doc.source_id),
                "created_at": doc.created_at.isoformat(),
                "metadata": doc.metadata
            }
            for doc in documents
        ]

    async def get_documents_by_job(self, job_id: UUID) -> List[Dict[str, Any]]:
        """Get all documents from a specific ingestion job"""
        documents = await self.document_repo.get_by_job(job_id)
        return [
            {
                "id": str(doc.id),
                "title": doc.title,
                "content": doc.content,
                "metadata": doc.metadata,
                "created_at": doc.created_at.isoformat()
            }
            for doc in documents
        ]

    async def delete_documents_by_job(self, job_id: UUID) -> int:
        """Delete all documents from a specific job"""
        return await self.document_repo.delete_by_job(job_id)