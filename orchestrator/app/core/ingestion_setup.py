from app.services.data_ingestion import (
    IngestionRegistry,
    LlamaIndexGoogleDriveIngestor,
    LlamaIndexNotionIngestor,
    LlamaIndexDatabaseIngestor
)
from app.repositories.data_source_repository import DataSourceRepository
from app.repositories.ingestion_job_repository import IngestionJobRepository
from app.repositories.document_repository import DocumentRepository

async def setup_ingestion_system(db_url: str) -> tuple:
    """
    Initialize the complete data ingestion system
    Returns: (data_source_repo, ingestion_job_repo, document_repo, ingestion_registry)
    """
    
    # Initialize repositories
    data_source_repo = DataSourceRepository(db_url)
    ingestion_job_repo = IngestionJobRepository(db_url)
    document_repo = DocumentRepository(db_url)
    
    # Create tables
    await data_source_repo.create_tables()
    await ingestion_job_repo.create_tables()
    await document_repo.create_tables()
    
    # Initialize ingestion registry
    registry = IngestionRegistry()
    
    # Register LlamaIndex ingestors
    registry.register(LlamaIndexGoogleDriveIngestor())
    registry.register(LlamaIndexNotionIngestor())
    registry.register(LlamaIndexDatabaseIngestor())
    
    return data_source_repo, ingestion_job_repo, document_repo, registry
