from app.core.security import encrypt_secret
from fastapi import HTTPException
from datetime import datetime
from app.core.config import settings
from app.core.security import decrypt_secret
from app.repositories.credential_repository import CredentialRepository
from app.models.credential_model import Credential
import httpx
from uuid import UUID

class CredentialService:
    def __init__(self, repository: CredentialRepository):
        self.repository = repository

    async def create_credential(self, provider: str, type_: str, data: dict, metadata: dict, created_by: UUID):
        encrypted_data = encrypt_secret(data)
        return await self.repository.create(provider, type_, encrypted_data, metadata, created_by)

    async def get_credential(self, credential_id: UUID):
        return await self.repository.get(credential_id)

    async def list_credentials(self):
        return await self.repository.list_all()
    
    async def list_credentials_by_user(self, user_id: UUID):
        return await self.repository.list_by_user(user_id)

    async def update_credential(self, credential_id: UUID, provider: str, type_: str, data: dict, metadata: dict):
        encrypted_data = encrypt_secret(data)
        return await self.repository.update(credential_id, provider, type_, encrypted_data, metadata)

    async def delete_credential(self, credential_id: UUID):
        return await self.repository.delete(credential_id)

    async def get_runtime_token(self, credential_id: UUID):
        """
        Retrieve a runtime token for a credential.
        - For OAuth: fetch and refresh using Auth Service.
        - For API keys or static credentials: return decrypted data directly.
        """
        credential = await self.repository.get(credential_id)
        if not credential:
            raise HTTPException(status_code=404, detail="Credential not found")

        decrypted_data = decrypt_secret(credential.encrypted_data)

        if credential.type == "oauth":
            return await self._handle_oauth_token(credential, decrypted_data)

        elif credential.type == "api_key":
            return {
                "credential_id": str(credential.id),
                "provider": credential.provider,
                "type": credential.type,
                "metadata": credential.cred_metadata or {},
                "runtime_token": decrypted_data.get("api_key"),
                "expires_at": None,
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported credential type: {credential.type}")

    async def _handle_oauth_token(self, credential: Credential, decrypted_data: dict):
        """Handles runtime token retrieval for OAuth credentials."""
        access_token = decrypted_data.get("access_token")
        expires_at = decrypted_data.get("expires_at")
        refresh_token = decrypted_data.get("refresh_token")

        # TODO: Consider implementing proactive token refresh (e.g. refresh if expiring in the next 5 minutes)
        if expires_at:
            # Handle both string and datetime objects
            if isinstance(expires_at, str):
                expires_datetime = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            else:
                expires_datetime = expires_at
            
            if datetime.utcnow() >= expires_datetime:
                refreshed_token = await self._refresh_token(str(credential.created_by), credential.provider)

            # Merge old token data to preserve fields like refresh_token or scopes
            merged_token = {
                **decrypted_data,      # keep old data
                **refreshed_token      # override with new token fields
            }
            
            # Encrypt the new token data
            encrypted_data = encrypt_secret(merged_token)
            await self.repository.update_encrypted_data(credential.id, encrypted_data)

            decrypted_data = merged_token
            access_token = refreshed_token["access_token"]
            expires_at = refreshed_token["expires_at"]

        return {
            "credential_id": str(credential.id),
            "provider": credential.provider,
            "type": credential.type,
            "metadata": credential.cred_metadata or {},
            "runtime_token": access_token,
            "scopes": decrypted_data.get("scopes", []),
            "expires_at": expires_at,
        }

    async def _refresh_token(self, user_id: str, provider: str):
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