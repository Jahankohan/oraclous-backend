from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Service Configuration
    SERVICE_NAME: str = "knowledge-graph-builder"
    SERVICE_VERSION: str = "1.0.0"
    SERVICE_URL: str = "http://localhost:8003"
    
    # Database Configuration
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"
    
    POSTGRES_URL: str = "postgresql+asyncpg://postgres:password@postgres:5432/kgbuilder"
    
    # External Services
    AUTH_SERVICE_URL: str = "http://auth-service:8000"
    CREDENTIAL_BROKER_URL: str = "http://credential-broker:8000"
    CORE_SERVICE_URL: str = "http://oraclous-core:8000"
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # Security
    INTERNAL_SERVICE_KEY: str = "your-internal-service-key"
    JWT_SECRET_KEY: str = "your-jwt-secret"
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DIFFBOT_API_KEY: Optional[str] = None

    # Add config flags to control these features
    ENABLE_SIMILARITY_PROCESSING: bool = True
    ENABLE_COMMUNITY_DETECTION: bool = True

    # Performance Settings
    MAX_CONCURRENT_EXTRACTIONS: int = 5
    BATCH_SIZE: int = 100
    CACHE_TTL: int = 300
    
    # Monitoring
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()