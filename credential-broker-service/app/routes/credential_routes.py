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
    user_id = UUID(request.state.user["id"])  # Ensure it's a UUID object
    return await service.create_credential(payload.provider, payload.type, payload.data, payload.cred_metadata, user_id)


@router.post("/internal/oauth-credentials")
async def create_oauth_credential(payload: OAuthCredentialCreate, request: Request, _=Depends(verify_internal_service)):
    """
    Internal endpoint for Auth Service to store OAuth credentials.
    - Requires x-internal-service-key for authentication.
    """
    logger.info(f"Creating OAuth credential for provider: {payload.provider} by user: {payload.created_by}")
    
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)

    # Ensure created_by is a UUID object
    created_by_uuid = UUID(payload.created_by) if isinstance(payload.created_by, str) else payload.created_by

    return await service.create_credential(
        provider=payload.provider,
        type_="oauth",
        data=payload.data,
        metadata=payload.cred_metadata,
        created_by=created_by_uuid
    )

@router.get("/{credential_id}")
async def get_credential(credential_id: UUID, request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    user_id = UUID(request.state.user["id"])  # Ensure it's a UUID object
    cred = await service.get_credential(credential_id)
    if not cred:
        raise HTTPException(status_code=404, detail="Credential not found")
    # Verify user owns this credential
    if str(cred.created_by) != str(user_id):
        raise HTTPException(status_code=403, detail="Access denied")
    return cred

@router.get("/", response_model=List[CredentialResponse])
async def list_credentials(request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    user_id = UUID(request.state.user["id"])  # Ensure it's a UUID object
    return await service.list_credentials_by_user(user_id)

@router.put("/{credential_id}", response_model=CredentialResponse)
async def update_credential(credential_id: UUID, payload: CredentialCreate, request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    user_id = UUID(request.state.user["id"])  # Ensure it's a UUID object
    
    # Verify user owns this credential
    existing_cred = await service.get_credential(credential_id)
    if not existing_cred:
        raise HTTPException(status_code=404, detail="Credential not found")
    if str(existing_cred.created_by) != str(user_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    updated_cred = await service.update_credential(credential_id, payload.provider, payload.type, payload.data, payload.cred_metadata)
    return updated_cred

@router.delete("/{credential_id}")
async def delete_credential(credential_id: UUID, request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    user_id = UUID(request.state.user["id"])  # Ensure it's a UUID object
    
    # Verify user owns this credential
    existing_cred = await service.get_credential(credential_id)
    if not existing_cred:
        raise HTTPException(status_code=404, detail="Credential not found")
    if str(existing_cred.created_by) != str(user_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    deleted = await service.delete_credential(credential_id)
    return {"message": "Credential deleted successfully"}

@router.get("/{credential_id}/runtime-token")
async def get_runtime_token(credential_id: UUID, request: Request):
    credential_repository = await get_credential_repository(request)
    service = CredentialService(credential_repository)
    user_id = UUID(request.state.user["id"])  # Ensure it's a UUID object
    
    # Verify user owns this credential
    existing_cred = await service.get_credential(credential_id)
    if not existing_cred:
        raise HTTPException(status_code=404, detail="Credential not found")
    if str(existing_cred.created_by) != str(user_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    runtime_token = await service.get_runtime_token(credential_id)
    return runtime_token
