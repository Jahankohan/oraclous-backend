from fastapi import APIRouter, HTTPException, Request, Depends
from uuid import UUID
from app.schemas.credential_schema import CredentialCreate, CredentialResponse, OAuthCredentialCreate
from app.services.credential_service import CredentialService
from app.core.dependencies import get_credential_repository, verify_internal_service
from typing import List
import logging


logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=CredentialResponse)
async def create_credential(payload: CredentialCreate, request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    user_id = request.state.user["id"]
    return await service.create_credential(payload.provider, payload.type, payload.data, payload.metadata, user_id)


@router.post("/internal/oauth-credentials", response_model=CredentialResponse)
async def create_oauth_credential(payload: OAuthCredentialCreate, request: Request):
    """
    Internal endpoint for Auth Service to store OAuth credentials.
    - Requires x-internal-service-key for authentication.
    """
    logger.info(f"Creating OAuth credential for provider: {payload.provider} by user: {payload.created_by}")
    
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)

    return await service.create_credential(
        provider=payload.provider,
        type_="oauth",
        data=payload.data,
        metadata=payload.metadata,
        created_by=payload.created_by  # Explicitly passed by Auth Service
    )

@router.get("/{credential_id}")
async def get_credential(credential_id: UUID, request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    cred = await service.get_credential(credential_id)
    if not cred:
        raise HTTPException(status_code=404, detail="Credential not found")
    return cred

@router.get("/", response_model=List[CredentialResponse])
async def list_credentials(request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    return await service.list_credentials()

@router.put("/{credential_id}", response_model=CredentialResponse)
async def update_credential(credential_id: UUID, payload: CredentialCreate, request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    updated_cred = await service.update_credential(credential_id, payload.provider, payload.type, payload.data)
    if not updated_cred:
        raise HTTPException(status_code=404, detail="Credential not found")
    return updated_cred

@router.delete("/{credential_id}")
async def delete_credential(credential_id: UUID, request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    deleted = await service.delete_credential(credential_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Credential not found")
    return {"message": "Credential deleted successfully"}

@router.get("/{credential_id}/runtime-token")
async def get_runtime_token(credential_id: UUID, request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    runtime_token = await service.get_runtime_token(credential_id)
    if not runtime_token:
        raise HTTPException(status_code=404, detail="Credential not found or invalid")
    return runtime_token
