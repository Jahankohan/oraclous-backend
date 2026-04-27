from fastapi import APIRouter

from app.api import schema
from app.api.v1.endpoints import (
    agents,
    chat,
    code_graphs,
    connectors,
    evaluation,
    federation,
    graphs,
    health,
    memories,
    multimodal,
    permissions,
    service_accounts,
    webhooks,
)
from app.api.v1.endpoints.federation import graph_federation_router

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(graphs.router, tags=["graphs"])
api_router.include_router(agents.router, prefix="/api/v1", tags=["agents"])
api_router.include_router(multimodal.router, tags=["multimodal"])
api_router.include_router(code_graphs.router, tags=["code-knowledge-graph"])
api_router.include_router(chat.router, prefix="/api/v1", tags=["chat"])
api_router.include_router(schema.router, prefix="/api/v1", tags=["schema"])
api_router.include_router(evaluation.router, prefix="/api/v1", tags=["evaluation"])
api_router.include_router(federation.router, prefix="/api/v1", tags=["federation"])
api_router.include_router(graph_federation_router, prefix="/api/v1", tags=["federation"])
api_router.include_router(permissions.router, prefix="/api/v1", tags=["permissions"])
api_router.include_router(memories.router, prefix="/api/v1", tags=["memories"])
api_router.include_router(connectors.router, prefix="/api/v1", tags=["connectors"])
api_router.include_router(webhooks.router, prefix="/api/v1", tags=["webhooks"])
api_router.include_router(
    service_accounts.router, prefix="/api/v1", tags=["service-accounts"]
)
