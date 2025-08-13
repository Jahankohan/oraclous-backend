from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any, Union
from enum import Enum

class DocumentSource(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    YOUTUBE = "youtube"
    WIKIPEDIA = "wiki"
    WEB = "web"

class ProcessingMode(str, Enum):
    FROM_BEGINNING = "from_beginning"
    DELETE_AND_RESTART = "delete_and_restart"
    FROM_LAST_POSITION = "from_last_position"

class ChatMode(str, Enum):
    VECTOR = "vector"
    GRAPH_VECTOR = "graph_vector"
    GRAPH = "graph"
    FULLTEXT = "fulltext"
    GRAPH_VECTOR_FULLTEXT = "graph_vector_fulltext"
    ENTITY_VECTOR = "entity_vector"
    GLOBAL_VECTOR = "global_vector"

class Neo4jConnectionRequest(BaseModel):
    uri: str = Field(..., description="Neo4j database URI")
    username: str = Field(..., description="Neo4j username")
    password: str = Field(..., description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")

class DocumentUploadRequest(BaseModel):
    source_type: DocumentSource
    content: Optional[str] = None
    file_name: Optional[str] = None
    url: Optional[HttpUrl] = None
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    gcs_project_id: Optional[str] = None
    gcs_bucket: Optional[str] = None
    gcs_blob_name: Optional[str] = None

class ExtractionRequest(BaseModel):
    file_names: List[str]
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    node_labels: Optional[List[str]] = None
    relationship_types: Optional[List[str]] = None
    enable_schema: bool = Field(default=True, description="Enable schema-guided extraction")

class GraphQueryRequest(BaseModel):
    file_names: Optional[List[str]] = None
    limit: int = Field(default=100, ge=1, le=1000)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    mode: ChatMode = ChatMode.GRAPH_VECTOR
    file_names: Optional[List[str]] = None
    session_id: Optional[str] = None

class SchemaRequest(BaseModel):
    text: str = Field(..., description="Text to generate schema from")
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")

class DuplicateNodesRequest(BaseModel):
    node_ids: List[str] = Field(..., description="Node IDs to merge")
    target_node_id: str = Field(..., description="Target node to merge into")
