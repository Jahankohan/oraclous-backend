"""LLMConfig CRUD endpoints (STORY-021).

Org-level and project-level LLM configurations. API keys are forwarded to the
credential-broker-service and never stored in Neo4j or returned in any response.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from app.api.dependencies import (
    get_current_user,
    get_current_user_id,
    verify_graph_access,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client
from app.schemas.llm_config_schemas import (
    LLMConfigCreate,
    LLMConfigCreateResponse,
    LLMConfigResponse,
)
from app.services.credential_broker_client import (
    CredentialBrokerClient,
    CredentialBrokerError,
)
from app.services.llm_config_service import LLMConfigService

router = APIRouter()
logger = get_logger(__name__)

_MASK = "••••"


def _mask_key(api_key: str) -> str:
    return _MASK + api_key[-4:] if len(api_key) >= 4 else _MASK


def _to_response(d: dict) -> LLMConfigResponse:
    return LLMConfigResponse(
        config_id=d["config_id"],
        scope=d["scope"],
        provider=d["provider"],
        model=d["model"],
        base_url=d.get("base_url"),
        api_version=d.get("api_version"),
        api_key_masked=_MASK + "????",  # api_key is never stored; masked placeholder
        created_at=d["created_at"],
        deactivated_at=d.get("deactivated_at"),
    )


def _llm_config_service() -> LLMConfigService:
    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )
    return LLMConfigService(neo4j_client.async_driver)


def _broker() -> CredentialBrokerClient:
    return CredentialBrokerClient(settings.CREDENTIAL_BROKER_URL)


async def _get_org_id(current_user: dict = Depends(get_current_user)) -> str:
    """Extract org_id from the JWT principal (tenant_id field)."""
    org_id = current_user.get("tenant_id") or current_user.get("org_id") or ""
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No org context in token",
        )
    return str(org_id)


# ── Org-level endpoints ────────────────────────────────────────────────────────


@router.post(
    "/org/llm-configs",
    response_model=LLMConfigCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an org-level LLM config",
    responses={
        400: {"description": "Credential-broker error"},
        403: {"description": "Access denied"},
        503: {"description": "Neo4j or credential-broker unavailable"},
    },
)
async def create_org_config(
    data: LLMConfigCreate,
    user_id: str = Depends(get_current_user_id),
    org_id: str = Depends(_get_org_id),
    svc: LLMConfigService = Depends(_llm_config_service),
    broker: CredentialBrokerClient = Depends(_broker),
):
    try:
        api_key_ref = await broker.store_api_key(
            api_key=data.api_key,
            label=f"llm-config-org-{org_id}-{data.provider.value}",
            user_id=user_id,
        )
    except CredentialBrokerError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    config_id = await svc.create_org_config(org_id, user_id, data, api_key_ref)
    return LLMConfigCreateResponse(config_id=config_id)


@router.get(
    "/org/llm-configs",
    response_model=list[LLMConfigResponse],
    summary="List org-level LLM configs",
    responses={403: {"description": "Access denied"}},
)
async def list_org_configs(
    org_id: str = Depends(_get_org_id),
    svc: LLMConfigService = Depends(_llm_config_service),
):
    configs = await svc.list_org_configs(org_id)
    return [_to_response(c) for c in configs]


@router.delete(
    "/org/llm-configs/{config_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Deactivate an org-level LLM config",
    responses={
        403: {"description": "Access denied"},
        404: {"description": "Config not found"},
    },
)
async def delete_org_config(
    config_id: str,
    org_id: str = Depends(_get_org_id),
    svc: LLMConfigService = Depends(_llm_config_service),
):
    deleted = await svc.deactivate_config(config_id, org_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="LLM config not found"
        )


# ── Project-level endpoints ────────────────────────────────────────────────────


@router.post(
    "/graphs/{graph_id}/llm-configs",
    response_model=LLMConfigCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a project-level LLM config",
    responses={
        400: {"description": "Credential-broker error"},
        403: {"description": "Access denied"},
        503: {"description": "Neo4j or credential-broker unavailable"},
    },
)
async def create_project_config(
    graph_id: str,
    data: LLMConfigCreate,
    user_id: str = Depends(get_current_user_id),
    org_id: str = Depends(_get_org_id),
    svc: LLMConfigService = Depends(_llm_config_service),
    broker: CredentialBrokerClient = Depends(_broker),
):
    await verify_graph_access(graph_id, "admin", user_id)
    try:
        api_key_ref = await broker.store_api_key(
            api_key=data.api_key,
            label=f"llm-config-project-{graph_id}-{data.provider.value}",
            user_id=user_id,
        )
    except CredentialBrokerError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    config_id = await svc.create_project_config(
        graph_id, org_id, user_id, data, api_key_ref
    )
    return LLMConfigCreateResponse(config_id=config_id)


@router.get(
    "/graphs/{graph_id}/llm-configs",
    response_model=list[LLMConfigResponse],
    summary="List project-level LLM configs",
    responses={403: {"description": "Access denied"}},
)
async def list_project_configs(
    graph_id: str,
    user_id: str = Depends(get_current_user_id),
    svc: LLMConfigService = Depends(_llm_config_service),
):
    await verify_graph_access(graph_id, "read", user_id)
    configs = await svc.list_project_configs(graph_id)
    return [_to_response(c) for c in configs]
