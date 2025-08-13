from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from app.config.settings import get_settings
from app.core.neo4j_client import Neo4jClient
from app.routers import infrastructure, documents, graph, chat
from app.core.exceptions import Neo4jConnectionError, ServiceError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    logger.info("Starting Neo4j LLM Graph Builder Backend...")
    yield
    logger.info("Shutting down Neo4j LLM Graph Builder Backend...")

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
            content={"error": "Database connection failed", "detail": str(exc)}
        )

    @app.exception_handler(ServiceError)
    async def service_exception_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content={"error": "Service error", "detail": str(exc)}
        )

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "version": "2.0.0"}

    return app

app = create_app()