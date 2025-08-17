# --- Neo4j Write Test on Startup ---
import logging
from app.core.neo4j_client import get_neo4j_client
try:
    neo4j = get_neo4j_client()
    result = neo4j.execute_write_query("CREATE (t:TestNode {createdAt: datetime()}) RETURN t")
    logging.info(f"Neo4j write test successful: {result}")
except Exception as e:
    logging.error(f"Neo4j write test failed: {e}")

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from app.config.settings import get_settings
from app.core.neo4j_pool import get_neo4j_pool, close_neo4j_pool
from app.routers import infrastructure, documents, graph, chat
from app.core.exceptions import Neo4jConnectionError, ServiceError

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # or DEBUG if you want even more details
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    logger.info("Starting Neo4j LLM Graph Builder Backend...")
    
    # Initialize Neo4j connection pool
    try:
        pool = await get_neo4j_pool()
        await pool.ensure_constraints()
        
        # Create necessary indexes
        settings = get_settings()
        if settings.enable_embeddings:
            dimensions = 384  # Default for sentence transformers
            if settings.embedding_model.value == "text-embedding-ada-002":
                dimensions = 1536
            elif settings.embedding_model.value == "textembedding-gecko@003":
                dimensions = 768
            
            await pool.create_vector_index(dimensions=dimensions)
        
        # Create fulltext index
        await pool.create_fulltext_index()
        
        logger.info("Neo4j connection pool initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down Neo4j LLM Graph Builder Backend...")
    await close_neo4j_pool()
    logger.info("Cleanup completed")

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="Neo4j LLM Graph Builder API",
        description="Transform unstructured data into Neo4j knowledge graphs using LLMs",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware for tracing
    @app.middleware("http")
    async def add_request_id(request, call_next):
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Include routers
    app.include_router(infrastructure.router, prefix="/api/v1", tags=["Infrastructure"])
    app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
    app.include_router(graph.router, prefix="/api/v1", tags=["Graph"])
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])

    # Global exception handlers
    @app.exception_handler(Neo4jConnectionError)
    async def neo4j_connection_exception_handler(request, exc):
        return JSONResponse(
            status_code=503,
            content={
                "error": "Database connection failed",
                "detail": str(exc),
                "request_id": getattr(request.state, "request_id", None)
            }
        )

    @app.exception_handler(ServiceError)
    async def service_exception_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Service error",
                "detail": str(exc),
                "request_id": getattr(request.state, "request_id", None)
            }
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "detail": str(exc),
                "request_id": getattr(request.state, "request_id", None)
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None)
            }
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            pool = await get_neo4j_pool()
            db_health = await pool.health_check()
            
            return {
                "status": "healthy" if db_health["connected"] else "unhealthy",
                "version": "2.0.0",
                "database": db_health
            }
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "version": "2.0.0",
                    "error": str(e)
                }
            )

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Neo4j LLM Graph Builder API",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health"
        }

    return app

app = create_app()