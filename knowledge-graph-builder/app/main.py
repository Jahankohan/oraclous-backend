import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import create_tables
from app.core.errors import KGBError
from app.core.logging import get_logger, setup_logging
from app.core.neo4j_client import neo4j_client
from app.core.rate_limiter import limiter
from app.core.telemetry import (
    current_trace_context,
    instrument_fastapi,
    setup_telemetry,
    shutdown_telemetry,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info(f"Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")

    # Initialise OpenTelemetry (no-op when OTEL_ENABLED=false)
    setup_telemetry()

    try:
        # Initialize databases
        await create_tables()
        await neo4j_client.connect()
        neo4j_client.connect_sync()

        # Initialize ReBAC schema (Phase A + Phase B) + sync existing data
        from app.core.database import async_session_maker
        from app.services.rebac_service import rebac_service

        await rebac_service.initialize_schema(neo4j_client.async_driver)
        await rebac_service.initialize_schema_full(neo4j_client.async_driver)
        await rebac_service.seed_system_permissions(neo4j_client.async_driver)
        async with async_session_maker() as db:
            await rebac_service.sync_existing_data(neo4j_client.async_driver, db)

        # Ensure versioning + fingerprint indexes (idempotent)
        from app.services.snapshot_service import snapshot_service

        await snapshot_service.ensure_indexes()
        from app.services.pipeline_service import ensure_fingerprint_indexes

        await ensure_fingerprint_indexes()

        # Apply Code KG constraints and indexes (idempotent, IF NOT EXISTS)
        from app.services.code_parser_service import ensure_code_schema

        await ensure_code_schema(neo4j_client.async_driver)

        # Apply Assessment substrate constraints + indexes + catalog anchors
        # (STORY-026, TASK-067/TASK-069 — idempotent, safe on every boot)
        from app.db.assessment_schema_init import ensure_assessment_schema

        await ensure_assessment_schema(neo4j_client.async_driver)

        # Initialize AgentServiceAccount Neo4j schema (constraints + indexes)
        from app.services.service_account_service import service_account_service

        await service_account_service.initialize_schema(neo4j_client.async_driver)

        # Ensure Memory API Neo4j indexes (idempotent, IF NOT EXISTS)
        from app.services.memory_service import ensure_memory_indexes

        await ensure_memory_indexes()

        # Initialize Database Connector Neo4j constraints + indexes (ORA-77)
        from app.services.database_connector_service import database_connector_service

        await database_connector_service.ensure_constraints()

        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application")
    await neo4j_client.disconnect()
    shutdown_telemetry()


# Create FastAPI app
app = FastAPI(
    title="Knowledge Graph Builder",
    description="Transform unstructured data into knowledge graphs and query them using natural language",
    version=settings.SERVICE_VERSION,
    docs_url="/docs" if settings.LOG_LEVEL == "DEBUG" else None,
    redoc_url="/redoc" if settings.LOG_LEVEL == "DEBUG" else None,
    lifespan=lifespan,
)

# Attach FastAPI OTel instrumentation (no-op when OTEL_ENABLED=false)
instrument_fastapi(app)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Structured KGB-XXXX exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Map standard HTTP status codes to structured KGB-XXXX error codes where a
    canonical mapping exists.  All other status codes fall through to a plain
    ``{"detail": ...}`` response so that existing callers are not broken.
    """
    if exc.status_code == 404:
        code, msg = KGBError.GRAPH_NOT_FOUND
        return JSONResponse(
            status_code=404,
            content={
                "error_code": code,
                "message": msg,
                "detail": str(exc.detail),
                "docs_url": "/docs",
            },
        )
    if exc.status_code == 403:
        code, msg = KGBError.PERMISSION_DENIED
        return JSONResponse(
            status_code=403,
            content={"error_code": code, "message": msg},
        )
    if exc.status_code == 429:
        code, msg = KGBError.RATE_LIMIT_EXCEEDED
        headers = getattr(exc, "headers", None) or {}
        retry_after = headers.get("Retry-After", "0")
        return JSONResponse(
            status_code=429,
            headers=headers,
            content={
                "error_code": code,
                "message": msg,
                "retry_after": int(retry_after) if retry_after.isdigit() else 0,
            },
        )
    if exc.status_code == 503:
        code, msg = KGBError.NEO4J_UNAVAILABLE
        return JSONResponse(
            status_code=503,
            content={"error_code": code, "message": msg, "detail": str(exc.detail)},
        )
    # All other status codes — return plain detail without a KGB code.
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing + trace ID middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    trace_ctx = current_trace_context()
    if trace_ctx.get("trace_id"):
        response.headers["X-Trace-Id"] = trace_ctx["trace_id"]
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception on {request.url}: {exc}", exc_info=True)
    origin = request.headers.get("origin", "")
    cors_headers = {}
    if origin:
        cors_headers["Access-Control-Allow-Origin"] = origin
        cors_headers["Access-Control-Allow-Credentials"] = "true"
        cors_headers["Vary"] = "Origin"
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": str(request.url.path),
            "method": request.method,
        },
        headers=cors_headers,
    )


# Include API router
app.include_router(api_router, prefix="/api/v1")

# Public integration endpoints (integration-key auth, no JWT middleware)
from app.api.public.endpoints import public_agents as _public_agents
app.include_router(_public_agents.router, prefix="/public", tags=["public"])


# Root endpoint
@app.get("/")
async def root():
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
