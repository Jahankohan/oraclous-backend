from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List, Optional
from enum import Enum

class EmbeddingModel(str, Enum):
    SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"
    OPENAI = "text-embedding-ada-002" 
    VERTEX_AI = "textembedding-gecko@003"

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"

class Settings(BaseSettings):
    # Neo4j Configuration
    neo4j_uri: str = Field(default="neo4j://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")
    neo4j_user_agent: str = Field(default="llm-graph-builder", env="NEO4J_USER_AGENT")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    azure_openai_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    
    # Default Models
    default_llm_model: str = Field(default="gpt-4o-mini", env="DEFAULT_LLM_MODEL")
    embedding_model: EmbeddingModel = Field(default=EmbeddingModel.SENTENCE_TRANSFORMER, env="EMBEDDING_MODEL")
    
    # Processing Configuration
    max_token_chunk_size: int = Field(default=10000, env="MAX_TOKEN_CHUNK_SIZE")
    duplicate_score_threshold: float = Field(default=0.97, env="DUPLICATE_SCORE_VALUE")
    duplicate_text_distance: int = Field(default=5, env="DUPLICATE_TEXT_DISTANCE")
    knn_min_score: float = Field(default=0.94, env="KNN_MIN_SCORE")
    chunks_to_combine: int = Field(default=5, env="NUMBER_OF_CHUNKS_TO_COMBINE")
    
    # Storage Configuration
    gcs_bucket_name: Optional[str] = Field(default=None, env="BUCKET")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-west-2", env="AWS_REGION")
    
    # API Configuration
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    api_rate_limit: int = Field(default=100, env="API_RATE_LIMIT")
    
    # Feature Flags
    enable_embeddings: bool = Field(default=True, env="IS_EMBEDDING")
    enable_user_agent: bool = Field(default=True, env="ENABLE_USER_AGENT")
    gcs_file_cache: bool = Field(default=False, env="GCS_FILE_CACHE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
