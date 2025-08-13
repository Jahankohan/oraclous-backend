from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    NEW = "New"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

class BaseResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class ConnectionResponse(BaseModel):
    status: str
    data: Dict[str, Any]

class DocumentInfo(BaseModel):
    id: str
    file_name: str
    source_type: str
    status: ProcessingStatus
    created_at: datetime
    processed_at: Optional[datetime] = None
    node_count: Optional[int] = None
    relationship_count: Optional[int] = None
    chunk_count: Optional[int] = None

class GraphNode(BaseModel):
    id: str
    labels: List[str]
    properties: Dict[str, Any]

class GraphRelationship(BaseModel):
    id: str
    type: str
    start_node_id: str
    end_node_id: str
    properties: Dict[str, Any]

class GraphVisualization(BaseModel):
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    
class ChatResponse(BaseModel):
    message: str
    sources: List[Dict[str, Any]]
    session_id: str
    response_time: float

class SchemaResponse(BaseModel):
    node_labels: List[str]
    relationship_types: List[str]
    properties: Dict[str, List[str]]

class DuplicateNode(BaseModel):
    id: str
    name: str
    labels: List[str]
    similarity_score: float

class ProcessingProgress(BaseModel):
    file_name: str
    status: ProcessingStatus
    progress_percentage: float
    chunks_processed: int
    total_chunks: int
    current_step: str
    error_message: Optional[str] = None
