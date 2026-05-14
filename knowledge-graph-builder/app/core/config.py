from datetime import timedelta

from pydantic_settings import BaseSettings


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

    # Integration layer
    PUBLIC_BASE_URL: str = (
        "http://localhost:8003"  # used to build endpoint_url in PublishAgentResponse
    )

    # LLM Configuration
    OPENAI_API_KEY: str | None = None
    LLM_API_KEY: str | None = (
        None  # generic env-var fallback (aliases OPENAI_API_KEY when set)
    )
    LLM_MODEL: str = "gpt-4o"
    # Cap completion length. Critical for LM Studio: prompt+max_tokens must
    # fit in the per-slot budget (n_ctx / n_parallel when kv_unified).
    LLM_MAX_TOKENS: int = 3000
    ANTHROPIC_API_KEY: str | None = None
    DIFFBOT_API_KEY: str | None = None
    EMBEDDING_MODEL: str | None = "text-embedding-3-large"

    # Modern Knowledge Graph Configuration
    USE_ENTITY_BASE_TYPE: bool = True  # Use __Entity__ instead of Entity
    ENTITY_BASE_LABEL: str = "__Entity__"  # Base label for all entities
    ENABLE_DOCUMENT_HIERARCHY: bool = True  # Document → Chunk → Entity structure

    # Embedding Configuration
    EMBED_ALL_NODE_TYPES: bool = True  # Embed Documents, Chunks, and Entities
    ENABLE_DOCUMENT_EMBEDDINGS: bool = True
    ENABLE_CHUNK_EMBEDDINGS: bool = True
    ENABLE_ENTITY_EMBEDDINGS: bool = True

    # Vector Index Configuration
    ENABLE_UNIFIED_ENTITY_INDEXES: bool = True  # Single __Entity__ index
    VECTOR_INDEX_DIMENSIONS: int = 512
    VECTOR_SIMILARITY_FUNCTION: str = "cosine"

    # Document Processing
    PRESERVE_CHUNK_ORDER: bool = True  # Maintain chunk sequence
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 400
    MAX_CHUNKS_PER_DOCUMENT: int = 1000

    # Entity Extraction
    CONNECT_ENTITIES_TO_CHUNKS: bool = True  # Entities → Chunks (not Documents)
    MAX_ENTITY_TYPES_PER_GRAPH: int = 20
    ENABLE_SCHEMA_EVOLUTION: bool = True

    # Community Detection Settings
    COMMUNITY_DETECTION_MIN_ENTITIES: int = 50
    COMMUNITY_DETECTION_CONCURRENCY: int = 3
    LLM_SUMMARY_CONCURRENCY: int = 5

    # Legacy flags (disabled for modern approach)
    ENABLE_COMMUNITY_DETECTION: bool = True

    # STORY-7: SIMILAR_TO edge generation. The historic
    # ``ENABLE_SIMILARITY_PROCESSING`` flag was set to True but never
    # consumed anywhere in the code — no SIMILAR_TO edges ever materialised.
    # ``similarity_service.build_similarities`` is now the explicit
    # entry point (via POST /graphs/{id}/similarity/build). This setting
    # gates whether the ingest pipeline should call that service after
    # entity deduplication. Default False so existing ingest performance
    # is unchanged; opt-in per deployment.
    SIMILARITY_AUTO_TRIGGER_ON_INGEST: bool = False

    # STORY-6: post-ingest entity dedup + relationship consolidation.
    # ``entity_dedup_service.deduplicate`` is the explicit entry point
    # (via POST /graphs/{id}/entities/deduplicate). This setting gates
    # whether the ingest pipeline should call it automatically after
    # the existing MultiTenantEntityDeduplicator runs. Default False —
    # opt-in so existing ingest performance is unchanged; the on-demand
    # path is fine for graphs that don't need automatic cleanup.
    ENTITY_DEDUP_AUTO_TRIGGER_ON_INGEST: bool = False

    # Performance Settings
    MAX_CONCURRENT_EXTRACTIONS: int = 5
    BATCH_SIZE: int = 100
    CACHE_TTL: int = 300

    # Ingest rate limit. Applied via slowapi @limiter.limit to the three
    # document-ingest endpoints (text, document, image). 60/min is the
    # comfortable human-paced default; raise for batch loads (e.g. 300+).
    MAX_INGEST_RPM: int = 60

    # Optimization Settings
    OPTIMIZATION_INTERVAL: timedelta = timedelta(
        hours=2
    )  # Run optimization every 2 hours

    # Code Knowledge Graph Settings
    CODE_LARGE_REPO_DEPTH_THRESHOLD: int = (
        5000  # Auto-switch to depth:file above this file count
    )
    CODE_EMBEDDING_BATCH_SIZE: int = 50  # Symbols per embedding batch

    # Embedding Processing
    DOCUMENT_EMBEDDING_BATCH_SIZE: int = 10
    CHUNK_EMBEDDING_BATCH_SIZE: int = 5
    ENTITY_EMBEDDING_BATCH_SIZE: int = 10

    # CORS
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5174",
        "http://localhost:5173",
        "http://localhost:8080",
    ]

    # Monitoring
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"

    # OpenTelemetry
    OTEL_ENABLED: bool = False
    OTEL_SERVICE_NAME: str = "knowledge-graph-builder"
    OTEL_SERVICE_VERSION: str = "1.0.0"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://jaeger:4317"
    OTEL_EXPORTER_OTLP_PROTOCOL: str = "grpc"  # "grpc" or "http/protobuf"
    LOG_FORMAT: str = (
        "text"  # "json" for structured JSON logs, "text" for human-readable
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
