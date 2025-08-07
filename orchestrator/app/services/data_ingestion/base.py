from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from enum import Enum
import uuid
from datetime import datetime

class IngestionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DataSource(BaseModel):
    id: str = None
    name: str
    type: str  # "google_drive", "notion", "database", etc.
    config: Dict[str, Any]
    credentials: Optional[Dict[str, Any]] = None
    data_metadata: Optional[Dict[str, Any]] = None
    
    def __init__(self, **data):
        if not data.get('id'):
            data['id'] = str(uuid.uuid4())
        super().__init__(**data)

class IngestionJob(BaseModel):
    id: str = None
    source_id: str
    status: IngestionStatus = IngestionStatus.PENDING
    config: Dict[str, Any] = {}
    documents_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    user_id: Optional[str] = None
    
    def __init__(self, **data):
        if not data.get('id'):
            data['id'] = str(uuid.uuid4())
        if not data.get('created_at'):
            data['created_at'] = datetime.utcnow()
        super().__init__(**data)

class IngestionResult(BaseModel):
    job_id: str
    documents: List[Dict[str, Any]]
    job_metadata: Dict[str, Any]
    errors: List[str] = []

class BaseIngestor(ABC):
    """Base class for all data ingestors"""
    
    def __init__(self, name: str, supported_types: List[str]):
        self.name = name
        self.supported_types = supported_types
    
    @abstractmethod
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate the configuration for this ingestor"""
        pass
    
    @abstractmethod
    async def test_connection(self, source: DataSource) -> bool:
        """Test if the connection to the data source works"""
        pass
    
    @abstractmethod
    async def ingest(self, source: DataSource, job: IngestionJob) -> IngestionResult:
        """Perform the actual data ingestion"""
        pass
    
    @abstractmethod
    async def get_available_resources(self, source: DataSource) -> List[Dict[str, Any]]:
        """List available resources/files/documents from the source"""
        pass

