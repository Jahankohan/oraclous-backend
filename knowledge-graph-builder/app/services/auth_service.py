import httpx
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class AuthService:
    """Service for handling authentication with auth-service"""
    
    def __init__(self):
        self.auth_service_url = settings.AUTH_SERVICE_URL
        self.timeout = 10.0
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token with auth service"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                verification_url = f"{self.auth_service_url}/me"
                logger.debug(f"Verifying token with auth service: {verification_url}")
                response = await client.get(
                    verification_url,
                    headers={"Authorization": f"Bearer {token}"}
                )

                if response.status_code == 200:
                    user_data = response.json()
                    logger.debug(f"Token verified for user: {user_data.get('id')}")
                    return user_data
                elif response.status_code == 401:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token"
                    )
                else:
                    logger.error(f"Auth service error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Authentication service unavailable"
                    )
        except httpx.TimeoutException:
            logger.error("Timeout while verifying token with auth service")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service timeout"
            )
        except httpx.RequestError as e:
            logger.error(f"Request error while verifying token: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable"
            )

auth_service = AuthService()
