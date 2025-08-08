import httpx

from fastapi import HTTPException
from datetime import datetime
from app.core.config import settings
from app.core.security import decrypt_secret, encrypt_secret
from app.repositories.credential_repository import CredentialRepository
from app.models.credential_model import UserCredential
from app.schema.credential_schema import CreateCredential, RequestCredentials, CredentialsUpdate, RequestCredentialsResponse, CredentialOut
from typing import List
from uuid import UUID

class CredentialService:
    def __init__(self, repository: CredentialRepository):
        self.repository = repository

    async def create_credential(self, credential: CreateCredential) -> CredentialOut:
        credential.credential = encrypt_secret(credential.credential)
        cred = await self.repository.create_credential(credential)
        cred_out = CredentialOut(
            id=cred.id,
            name=cred.name,
            provider=cred.provider,
            user_id=cred.user_id,
            tool_id=cred.tool_id,
            cred_type=cred.cred_type
        )
        return cred_out

    async def get_credential_by_id(self, credential_id: UUID) -> RequestCredentialsResponse:
        credential = await self.repository.get_credential_by_id(credential_id)
        if credential:
            return await self.prepare_credential_out(credential)
        raise HTTPException(status_code=404, detail="Credential not found")

    async def list_credentials(self, cred_request: RequestCredentials) -> List[RequestCredentialsResponse]:
        credentials = await self.repository.list_credentials(cred_request)
        if credentials is None:
            raise HTTPException(status_code=404, detail="Credential not found")
        response = []
        for cred in credentials:
            cred_schema = await self.prepare_credential_out(cred)
            response.append(cred_schema)
        return response

    async def update_credential(self, credential_update: CredentialsUpdate) -> CredentialOut:
        credential_update.credential = encrypt_secret(credential_update.credential)
        cred = await self.repository.update_credential(credential_update)
        cred_out = CredentialOut(
            id=cred.id,
            name=cred.name,
            provider=cred.provider,
            user_id=cred.user_id,
            tool_id=cred.tool_id,
            cred_type=cred.cred_type
        )
        return cred_out

    async def delete_credential(self, credential_id: UUID) -> bool:
        return await self.repository.delete_credential(credential_id)
    
    async def prepare_credential_out(self, credential: UserCredential) -> RequestCredentialsResponse:
        cred_data = decrypt_secret(credential.encrypted_cred)
        credential_response = RequestCredentialsResponse(
            id=credential.id,
            name=credential.name,
            provider=credential.provider,
            user_id=credential.user_id,
            tool_id=credential.tool_id,
            cred_type=credential.cred_type,
            credential=cred_data
        )
        return credential_response

    # async def get_runtime_token(self, credential_id: UUID):
    #     """
    #     Retrieve a runtime token for a credential.
    #     - For OAuth: fetch and refresh using Auth Service.
    #     - For API keys or static credentials: return decrypted data directly.
    #     """
    #     credential = await self.repository.get(credential_id)
    #     if not credential:
    #         raise HTTPException(status_code=404, detail="Credential not found")

    #     decrypted_data = decrypt_secret(credential.encrypted_data)

    #     if credential.type == "oauth":
    #         return await self._handle_oauth_token(credential, decrypted_data)

    #     elif credential.type == "api_key":
    #         return {
    #             "credential_id": str(credential.id),
    #             "provider": credential.provider,
    #             "type": credential.type,
    #             "metadata": credential.cred_metadata or {},
    #             "runtime_token": decrypted_data.get("api_key"),
    #             "expires_at": None,
    #         }

    #     else:
    #         raise HTTPException(status_code=400, detail=f"Unsupported credential type: {credential.type}")

    # async def _handle_oauth_token(self, credential: Credential, decrypted_data: dict):
    #     """Handles runtime token retrieval for OAuth credentials."""
    #     access_token = decrypted_data.get("access_token")
    #     expires_at = decrypted_data.get("expires_at")
    #     refresh_token = decrypted_data.get("refresh_token")

    #     # TODO: Consider implementing proactive token refresh (e.g. refresh if expiring in the next 5 minutes)
    #     if expires_at:
    #         # Handle both string and datetime objects
    #         if isinstance(expires_at, str):
    #             expires_datetime = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
    #         else:
    #             expires_datetime = expires_at
            
    #         if datetime.utcnow() >= expires_datetime:
    #             refreshed_token = await self._refresh_token(str(credential.created_by), credential.provider)

    #         # Merge old token data to preserve fields like refresh_token or scopes
    #         merged_token = {
    #             **decrypted_data,      # keep old data
    #             **refreshed_token      # override with new token fields
    #         }
            
    #         # Encrypt the new token data
    #         encrypted_data = encrypt_secret(merged_token)
    #         await self.repository.update_encrypted_data(credential.id, encrypted_data)

    #         decrypted_data = merged_token
    #         access_token = refreshed_token["access_token"]
    #         expires_at = refreshed_token["expires_at"]

    #     return {
    #         "credential_id": str(credential.id),
    #         "provider": credential.provider,
    #         "type": credential.type,
    #         "metadata": credential.cred_metadata or {},
    #         "runtime_token": access_token,
    #         "scopes": decrypted_data.get("scopes", []),
    #         "expires_at": expires_at,
    #     }

    # async def _refresh_token(self, user_id: str, provider: str):
        """Calls Auth Service to refresh token."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.AUTH_SERVICE_URL}/oauth/runtime-tokens",
                params={"user_id": user_id, "provider": provider},
                headers={"x-internal-service-key": settings.INTERNAL_SERVICE_KEY}
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to refresh OAuth token")
            return response.json()