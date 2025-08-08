import httpx
import logging
from typing import List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass
from uuid import UUID
from app.core.constants import DATA_SOURCE_CAPABILITIES, OAUTH_ERROR_CODES

logger = logging.getLogger(__name__)

@dataclass
class TokenInfo:
    access_token: str
    expires_at: Optional[datetime]
    scopes: List[str]
    provider: str
    user_id: UUID

@dataclass
class CredentialResult:
    success: bool
    token: Optional[TokenInfo] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    login_url: Optional[str] = None
    missing_scopes: List[str] = None


class CredentialBroker:
    """
    Handles credential management and communication with auth service.
    This class should be used by the orchestrator to obtain OAuth tokens.
    """
    
    def __init__(self, auth_service_url: str, internal_service_key: str):
        self.auth_service_url = auth_service_url.rstrip('/')
        self.internal_service_key = internal_service_key
        self.headers = {
            "X-Internal-Key": self.internal_service_key,
            "Content-Type": "application/json"
        }

    async def get_provider_token(
        self, 
        user_id: UUID, 
        provider: str, 
        required_scopes: Optional[List[str]] = None
    ) -> CredentialResult:
        """
        Get a valid token for a provider, ensuring it has required scopes.
        Will attempt refresh if token is expired.
        """
        try:
            # First, validate current scopes
            if required_scopes:
                scope_result = await self._validate_scopes(user_id, provider, required_scopes)
                if not scope_result.success:
                    return scope_result

            # Get the actual token
            token_result = await self._get_runtime_token(user_id, provider)
            if not token_result.success:
                return token_result

            logger.info(f"Successfully obtained token for user {user_id}, provider {provider}")
            return token_result
            
        except Exception as e:
            logger.error(f"Unexpected error getting provider token: {e}")
            return CredentialResult(
                success=False,
                error_code=OAUTH_ERROR_CODES["PROVIDER_ERROR"],
                error_message=f"Failed to obtain token: {str(e)}"
            )

    async def ensure_data_source_access(
        self, 
        user_id: UUID, 
        provider: str, 
        data_source_type: str
    ) -> CredentialResult:
        """
        Ensure user has access to a specific data source type (e.g., 'drive', 'databases').
        Uses predefined capability mappings to determine required scopes.
        """
        try:
            # Get required scopes for this data source
            capabilities = DATA_SOURCE_CAPABILITIES.get(provider, {}).get(data_source_type)
            if not capabilities:
                return CredentialResult(
                    success=False,
                    error_code=OAUTH_ERROR_CODES["INVALID_PROVIDER"],
                    error_message=f"Unsupported data source: {provider}/{data_source_type}"
                )

            required_scopes = capabilities.get("required_scopes", [])
            return await self.get_provider_token(user_id, provider, required_scopes)
            
        except Exception as e:
            logger.error(f"Error ensuring data source access: {e}")
            return CredentialResult(
                success=False,
                error_code=OAUTH_ERROR_CODES["PROVIDER_ERROR"],
                error_message=str(e)
            )

    async def list_user_providers(self, user_id: UUID) -> Dict[str, List[str]]:
        """Get all providers and their scopes for a user."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.auth_service_url}/oauth/user-tokens",
                    headers=self.headers,
                    params={"user_id": user_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    result = {}
                    for provider_info in data["providers"]:
                        result[provider_info["provider"]] = provider_info["scopes"]
                    return result
                else:
                    logger.error(f"Failed to list user providers: {response.status_code}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error listing user providers: {e}")
            return {}

    async def get_available_data_sources(self, user_id: UUID) -> Dict[str, List[str]]:
        """
        Get available data sources for a user based on their current OAuth tokens.
        Returns mapping of provider -> list of available data source types.
        """
        try:
            user_providers = await self.list_user_providers(user_id)
            available_sources = {}
            
            for provider, scopes in user_providers.items():
                if provider in DATA_SOURCE_CAPABILITIES:
                    available_types = []
                    provider_capabilities = DATA_SOURCE_CAPABILITIES[provider]
                    
                    for source_type, config in provider_capabilities.items():
                        required_scopes = set(config.get("required_scopes", []))
                        user_scopes = set(scopes)
                        
                        # Check if user has all required scopes for this data source
                        if required_scopes.issubset(user_scopes):
                            available_types.append(source_type)
                    
                    if available_types:
                        available_sources[provider] = available_types
            
            return available_sources
            
        except Exception as e:
            logger.error(f"Error getting available data sources: {e}")
            return {}

    async def _validate_scopes(
        self, 
        user_id: UUID, 
        provider: str, 
        required_scopes: List[str]
    ) -> CredentialResult:
        """Validate that user has required scopes."""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "user_id": user_id,
                    "provider": provider,
                    "required_scopes": required_scopes
                }
                
                response = await client.post(
                    f"{self.auth_service_url}/oauth/validate-scopes",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data["valid"]:
                        return CredentialResult(success=True)
                    else:
                        return CredentialResult(
                            success=False,
                            error_code=OAUTH_ERROR_CODES["INSUFFICIENT_SCOPES"],
                            error_message="Insufficient scopes",
                            login_url=data.get("login_url"),
                            missing_scopes=data["missing_scopes"]
                        )
                else:
                    return CredentialResult(
                        success=False,
                        error_code=OAUTH_ERROR_CODES["PROVIDER_ERROR"],
                        error_message=f"Scope validation failed: {response.status_code}"
                    )
                    
        except Exception as e:
            logger.error(f"Error validating scopes: {e}")
            return CredentialResult(
                success=False,
                error_code=OAUTH_ERROR_CODES["PROVIDER_ERROR"],
                error_message=str(e)
            )

    async def _get_runtime_token(self, user_id: UUID, provider: str) -> CredentialResult:
        """Get runtime token from auth service."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.auth_service_url}/oauth/runtime-tokens",
                    headers=self.headers,
                    params={"user_id": user_id, "provider": provider}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    token_info = TokenInfo(
                        access_token=data["access_token"],
                        expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                        scopes=data["scopes"],
                        provider=provider,
                        user_id=user_id
                    )
                    return CredentialResult(success=True, token=token_info)
                    
                elif response.status_code == 404:
                    return CredentialResult(
                        success=False,
                        error_code=OAUTH_ERROR_CODES["TOKEN_NOT_FOUND"],
                        error_message="No token found for this provider"
                    )
                    
                elif response.status_code == 401:
                    return CredentialResult(
                        success=False,
                        error_code=OAUTH_ERROR_CODES["TOKEN_EXPIRED"],
                        error_message="Token expired and refresh failed"
                    )
                    
                else:
                    return CredentialResult(
                        success=False,
                        error_code=OAUTH_ERROR_CODES["PROVIDER_ERROR"],
                        error_message=f"Failed to get token: {response.status_code}"
                    )
                    
        except Exception as e:
            logger.error(f"Error getting runtime token: {e}")
            return CredentialResult(
                success=False,
                error_code=OAUTH_ERROR_CODES["PROVIDER_ERROR"],
                error_message=str(e)
            )

    async def refresh_token_if_needed(self, user_id: UUID, provider: str) -> CredentialResult:
        """Explicitly refresh a token if it's expired."""
        try:
            async with httpx.AsyncClient() as client:
                payload = {"user_id": user_id, "provider": provider}
                
                response = await client.post(
                    f"{self.auth_service_url}/oauth/refresh-if-needed",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data["success"]:
                        token_info = TokenInfo(
                            access_token=data["access_token"],
                            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                            scopes=[],  # Scopes not returned in refresh response
                            provider=provider,
                            user_id=user_id
                        )
                        return CredentialResult(success=True, token=token_info)
                    else:
                        return CredentialResult(
                            success=False,
                            error_code=OAUTH_ERROR_CODES["REFRESH_FAILED"],
                            error_message=data.get("error", "Token refresh failed")
                        )
                else:
                    return CredentialResult(
                        success=False,
                        error_code=OAUTH_ERROR_CODES["PROVIDER_ERROR"],
                        error_message=f"Refresh request failed: {response.status_code}"
                    )
                    
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return CredentialResult(
                success=False,
                error_code=OAUTH_ERROR_CODES["PROVIDER_ERROR"],
                error_message=str(e)
            )
