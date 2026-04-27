from datetime import UTC, datetime

from fastapi import APIRouter

from app.core.config import settings
from app.core.database import check_db_health
from app.core.neo4j_client import neo4j_client
from app.schemas.graph_schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["health"],
)
async def health_check():
    """
    Return the health status of the service and its dependencies.

    Does not require authentication. Use this endpoint for liveness and
    readiness probes in container orchestration. Returns `status: degraded`
    if Neo4j or PostgreSQL is unreachable.
    """

    # Check Neo4j
    neo4j_health = await neo4j_client.health_check()

    # Check PostgreSQL
    postgres_health = await check_db_health()

    # Determine overall status
    overall_status = "healthy"
    if neo4j_health["status"] != "healthy" or postgres_health["status"] != "healthy":
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        timestamp=datetime.now(UTC),
        dependencies={"neo4j": neo4j_health, "postgres": postgres_health},
    )
