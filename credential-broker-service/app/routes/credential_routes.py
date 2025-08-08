from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from uuid import UUID
from app.schema.credential_schema import CreateCredential, RequestCredentials, RequestCredentialsResponse, CredentialOut, CredentialsUpdate
from app.repositories.credential_repository import CredentialRepository
from app.services.credential_service import CredentialService
from fastapi.security import OAuth2PasswordBearer


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

router = APIRouter()

# Dependency to get repository from app state
from fastapi import Request
async def get_credential_repo(request: Request) -> CredentialRepository:
    return request.app.state.credential_repository

@router.post("/", response_model=CredentialOut)
async def create_credential(credential: CreateCredential, repo: CredentialRepository = Depends(get_credential_repo)):
    # ToDo: validate user exists and the same in the credentials
    credential_service = CredentialService(repo)
    cred = await credential_service.create_credential(credential)
    return cred

@router.get("/{cred_id}", response_model=RequestCredentialsResponse)
async def get_credential(cred_id: UUID, repo: CredentialRepository = Depends(get_credential_repo)):
    credential_service = CredentialService(repo)
    cred_response = await credential_service.get_credential_by_id(cred_id)
    return cred_response

@router.post("/retrieve/", response_model=List[RequestCredentialsResponse])
async def list_credentials(credential_request: RequestCredentials, repo: CredentialRepository = Depends(get_credential_repo)):
    credential_service = CredentialService(repo)
    cred_response = await credential_service.list_credentials(credential_request)
    return cred_response

@router.put("/{cred_id}", response_model=CredentialOut)
async def update_credential(cred_update: CredentialsUpdate, repo: CredentialRepository = Depends(get_credential_repo)):
    credential_service = CredentialService(repo)
    cred_response = await credential_service.update_credential(cred_update)
    return cred_response

@router.delete("/{cred_id}")
async def delete_credential(cred_id: UUID, repo: CredentialRepository = Depends(get_credential_repo)):
    credential_service = CredentialService(repo)
    ok = await credential_service.delete_credential(cred_id)
    if ok:
        return {"message": "Credential deleted successfully"}
