# app/services/credential_client.py
import httpx
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class CredentialClient:
    """
    Client for communicating with the Credential Broker Service
    Handles all credential-related operations for the orchestrator
    """
    def __init__(self):
        self.credential_broker_url = settings.CREDENTIAL_BROKER_URL
        self.internal_service_key = settings.INTERNAL_SERVICE_KEY
        self.headers = {
            "X-Internal-Key": self.internal_service_key,
            "Content-Type": "application/json"
        }
        self.timeout = 30.0

    async def get_runtime_token(
        self,
        user_id: UUID,
        provider: str,
        required_scopes: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get OAuth runtime token for a user and provider from credential broker
        """
        payload = {
            "user_id": str(user_id),
            "provider": provider,
            "required_scopes": required_scopes or []
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.credential_broker_url}/runtime-token",
                    headers=self.headers,
                    json=payload
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to get runtime token: {response.status_code} - {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Error getting runtime token: {e}")
            return None

    async def validate_credentials(
        self,
        user_id: UUID,
        credential_mappings: Dict[str, str],
        required_scopes: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate that all required credentials are available and valid
        Uses /ensure-data-source-access for OAuth, and GET for other credentials
        """
        validation_results = {}
        for cred_type, cred_identifier in credential_mappings.items():
            try:
                # Handle OAuth tokens
                if cred_type == "OAUTH_TOKEN":
                    scopes = required_scopes.get(cred_type, []) if required_scopes else []
                    # Use ensure_data_source_access for OAuth validation/flow
                    result = await self.ensure_data_source_access(user_id, cred_identifier, scopes)
                    if result.get("success"):
                        validation_results[cred_type] = {
                            "valid": True,
                            "error": None,
                            "login_url": None
                        }
                    else:
                        validation_results[cred_type] = {
                            "valid": False,
                            "error": result.get("error_message"),
                            "login_url": result.get("login_url"),
                            "missing_scopes": result.get("missing_scopes", [])
                        }
                else:
                    result = await self._validate_credential(cred_identifier)
                    validation_results[cred_type] = result
            except Exception as e:
                validation_results[cred_type] = {
                    "valid": False,
                    "error": str(e),
                    "error_code": "VALIDATION_ERROR"
                }
        return validation_results
    
    async def _get_credential_data(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Get credential data by ID from credential broker"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.credential_broker_url}/{credential_id}",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Return the actual credential data (API keys, connection strings, etc.)
                    return data.get("data", {})
                else:
                    logger.error(f"Failed to get credential: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting credential data: {e}")
            return None
    
    async def _validate_oauth_token(
        self, 
        user_id: UUID, 
        provider: str, 
        required_scopes: List[str]
    ) -> Dict[str, Any]:
        """Validate OAuth token without retrieving it"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "user_id": str(user_id),
                    "provider": provider,
                    "required_scopes": required_scopes
                }
                
                response = await client.post(
                    f"{self.credential_broker_url}/oauth/validate-scopes",
                    headers=self.headers,
                    json=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "valid": data.get("valid", False),
                        "error": None if data.get("valid") else "Invalid or insufficient scopes",
                        "missing_scopes": data.get("missing_scopes", []),
                        "login_url": data.get("login_url")
                    }
                else:
                    return {
                        "valid": False,
                        "error": f"Validation failed: {response.status_code}",
                        "error_code": "VALIDATION_FAILED"
                    }
                    
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "error_code": "VALIDATION_ERROR"
            }
    
    async def _validate_credential(self, credential_id: str) -> Dict[str, Any]:
        """Validate non-OAuth credential"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.credential_broker_url}/{credential_id}",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Check if credential has expired (if applicable)
                    expires_at = data.get("expires_at")
                    if expires_at:
                        expiry_time = datetime.fromisoformat(expires_at)
                        if expiry_time <= datetime.utcnow():
                            return {
                                "valid": False,
                                "error": "Credential has expired",
                                "error_code": "CREDENTIAL_EXPIRED"
                            }
                    
                    return {
                        "valid": True,
                        "error": None
                    }
                elif response.status_code == 404:
                    return {
                        "valid": False,
                        "error": "Credential not found",
                        "error_code": "CREDENTIAL_NOT_FOUND"
                    }
                else:
                    return {
                        "valid": False,
                        "error": f"Validation failed: {response.status_code}",
                        "error_code": "VALIDATION_FAILED"
                    }
                    
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "error_code": "VALIDATION_ERROR"
            }
    
    async def get_available_data_sources(self, user_id: UUID) -> Dict[str, List[str]]:
        """Get available data sources for user"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.credential_broker_url}/available-data-sources",
                    headers=self.headers,
                    params={"user_id": str(user_id)}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("available_data_sources", {})
                else:
                    logger.error(f"Failed to get available data sources: {response.status_code}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting available data sources: {e}")
            return {}
    
    async def ensure_data_source_access(
        self, 
        user_id: UUID, 
        provider: str, 
        required_scopes: List[str]
    ) -> Dict[str, Any]:
        """Ensure user has access to specific data source"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "user_id": str(user_id),
                    "provider": provider,
                    "required_scopes": required_scopes
                }
                
                response = await client.post(
                    f"{self.credential_broker_url}/ensure-data-source-access",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "access_token": data.get("access_token"),
                        "expires_at": data.get("expires_at"),
                        "scopes": data.get("scopes", [])
                    }
                else:
                    error_data = response.json() if response.status_code == 400 else {}
                    return {
                        "success": False,
                        "error_code": error_data.get("error_code"),
                        "error_message": error_data.get("error_message"),
                        "login_url": error_data.get("login_url"),
                        "missing_scopes": error_data.get("missing_scopes", [])
                    }
                    
        except Exception as e:
            logger.error(f"Error ensuring data source access: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "error_code": "CLIENT_ERROR"
            }
