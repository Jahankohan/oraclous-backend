from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from uuid import UUID
from app.schema.credential_schema import CreateCredential, RequestCredentials, RequestCredentialsResponse, CredentialOut, CredentialsUpdate, CredentialBrokerSuccessResponse, CredentialBrokerErrorResponse, ProvidersResponse, AvailableDataSourcesResponse, RuntimeTokenRequest
from app.repositories.credential_repository import CredentialRepository
from app.services.credential_service import CredentialService
from app.services.credential_broker_service import CredentialBroker
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

@router.post("/runtime-token", response_model=CredentialBrokerSuccessResponse | CredentialBrokerErrorResponse)
async def get_runtime_token(
    request: Request,
    request_data: RuntimeTokenRequest
):
    broker = CredentialBroker()
    result = await broker.get_provider_token(request_data.user_id, request_data.provider, request_data.required_scopes)
    if result.success:
        return CredentialBrokerSuccessResponse(
            access_token=result.token.access_token,
            expires_at=result.token.expires_at,
            scopes=result.token.scopes,
            provider=result.token.provider,
            user_id=result.token.user_id,
        )
    else:
        return CredentialBrokerErrorResponse(
            error_code=result.error_code,
            error_message=result.error_message,
            login_url=result.login_url,
            missing_scopes=result.missing_scopes,
        )

@router.get("/providers", response_model=ProvidersResponse)
async def list_user_providers(
    request: Request,
    user_id: UUID = Query(...)
):
    broker = CredentialBroker()
    providers = await broker.list_user_providers(user_id)
    return ProvidersResponse(user_id=user_id, providers=providers)

@router.get("/available-data-sources", response_model=AvailableDataSourcesResponse)
async def get_available_data_sources(
    request: Request,
    user_id: UUID = Query(...)
):
    broker = CredentialBroker()
    sources = await broker.get_available_data_sources(user_id)
    return AvailableDataSourcesResponse(user_id=user_id, available_data_sources=sources)

@router.post("/ensure-data-source-access", response_model=CredentialBrokerSuccessResponse | CredentialBrokerErrorResponse)
async def ensure_data_source_access(
    request: Request,
    request_data: RuntimeTokenRequest
):
    broker = CredentialBroker()
    result = await broker.ensure_data_source_access(request_data.user_id, request_data.provider, request_data.data_source_type)
    if result.success:
        return CredentialBrokerSuccessResponse(
            access_token=result.token.access_token,
            expires_at=result.token.expires_at,
            scopes=result.token.scopes,
            provider=result.token.provider,
            user_id=result.token.user_id,
        )
    else:
        return CredentialBrokerErrorResponse(
            error_code=result.error_code,
            error_message=result.error_message,
            login_url=result.login_url,
            missing_scopes=result.missing_scopes,
        )
