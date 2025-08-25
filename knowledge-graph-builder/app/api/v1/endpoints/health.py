from fastapi import APIRouter
from datetime import datetime
from app.core.config import settings
from app.core.neo4j_client import neo4j_client
from app.core.database import check_db_health
from app.schemas.graph_schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    # Check Neo4j
    neo4j_health = await neo4j_client.health_check()
    
    # Check PostgreSQL
    postgres_health = await check_db_health()
    
    # Determine overall status
    overall_status = "healthy"
    if (neo4j_health["status"] != "healthy" or 
        postgres_health["status"] != "healthy"):
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        timestamp=datetime.utcnow(),
        dependencies={
            "neo4j": neo4j_health,
            "postgres": postgres_health
        }
    )
