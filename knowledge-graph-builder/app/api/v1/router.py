from fastapi import APIRouter
from app.api.v1.endpoints import graphs, health, entities, diffbot, search, embeddings, chat

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(graphs.router, tags=["graphs"])
api_router.include_router(entities.router, tags=["entities"])
api_router.include_router(diffbot.router, tags=["diffbot"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(embeddings.router, tags=["embeddings"])
api_router.include_router(chat.router, tags=["chat"])
