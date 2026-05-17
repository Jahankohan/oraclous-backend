from fastapi import APIRouter

from app.api import schema
from app.api.v1.endpoints import (
    agents,
    chat,
    chat_history,
    code_graphs,
    communities,
    connectors,
    evaluation,
    federation,
    graphs,
    health,
    integration,
    llm_configs,
    memories,
    multimodal,
    organizations,
    permissions,
    service_accounts,
    webhooks,
)
from app.api.v1.endpoints.federation import graph_federation_router

api_router = APIRouter()

# Include all endpoint routers. The /api/v1 prefix is applied once by
# main.py:app.include_router(api_router, prefix="/api/v1"); per-router
# include_router calls must not pass prefix="/api/v1" or the path will
# be doubled. See TASK-090.
api_router.include_router(health.router, tags=["health"])
api_router.include_router(graphs.router, tags=["graphs"])
api_router.include_router(communities.router, tags=["communities"])
api_router.include_router(agents.router, tags=["agents"])
api_router.include_router(llm_configs.router, tags=["llm-configs"])
api_router.include_router(integration.router, tags=["integration"])
api_router.include_router(multimodal.router, tags=["multimodal"])
api_router.include_router(code_graphs.router, tags=["code-knowledge-graph"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(chat_history.router, tags=["chat-history"])
api_router.include_router(schema.router, tags=["schema"])
api_router.include_router(evaluation.router, tags=["evaluation"])
api_router.include_router(federation.router, tags=["federation"])
api_router.include_router(graph_federation_router, tags=["federation"])
api_router.include_router(permissions.router, tags=["permissions"])
api_router.include_router(memories.router, tags=["memories"])
api_router.include_router(connectors.router, tags=["connectors"])
api_router.include_router(organizations.router, tags=["organizations"])
api_router.include_router(webhooks.router, tags=["webhooks"])
api_router.include_router(service_accounts.router, tags=["service-accounts"])
