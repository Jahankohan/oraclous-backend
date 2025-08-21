import httpx
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class CredentialService:
    """Service for managing credentials via credential-broker"""
    
    def __init__(self):
        self.broker_url = settings.CREDENTIAL_BROKER_URL
        self.internal_key = settings.INTERNAL_SERVICE_KEY
        self.timeout = 10.0
    
    async def get_user_credentials(self, user_id: str, provider: str) -> Optional[Dict[str, Any]]:
        """Get user credentials for a specific provider"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.broker_url}/api/v1/runtime-token",
                    headers={"X-Internal-Key": self.internal_key},
                    json={"user_id": user_id, "provider": provider}
                )
                
                if response.status_code == 200:
                    credentials = response.json()
                    logger.debug(f"Retrieved credentials for user {user_id}, provider {provider}")
                    return credentials
                elif response.status_code == 404:
                    logger.warning(f"No credentials found for user {user_id}, provider {provider}")
                    return None
                else:
                    logger.error(f"Credential service error: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Error retrieving credentials: {e}")
            return None
    
    async def get_openai_token(self, user_id: str) -> Optional[str]:
        """Get OpenAI API token for user"""
        credentials = await self.get_user_credentials(user_id, "openai")
        return credentials.get("access_token") if credentials else None
    
    async def get_anthropic_token(self, user_id: str) -> Optional[str]:
        """Get Anthropic API token for user"""
        credentials = await self.get_user_credentials(user_id, "anthropic")
        return credentials.get("access_token") if credentials else None

credential_service = CredentialService()
