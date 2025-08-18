from app.core.config import settings
from fastapi import APIRouter

from app.api.v1.endpoints import tools, instances, workflow_routes

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(tools.router, prefix="/tools", tags=["tools"])
api_router.include_router(instances.router, prefix="/instances", tags=["instances"])
api_router.include_router(workflow_routes.router, prefix="/workflows", tags=["workflows"])

# Health check endpoint
@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}
