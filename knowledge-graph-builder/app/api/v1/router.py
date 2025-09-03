from fastapi import APIRouter
from app.api.v1.endpoints import graphs

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(graphs.router, tags=["graphs"])
