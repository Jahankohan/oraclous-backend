import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.services.auth_service import AuthService
from app.services.credential_service import CredentialService

@pytest.mark.asyncio
async def test_auth_service_verify_token_success():
    """Test successful token verification"""
    auth_service = AuthService()
    
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "user123",
        "email": "test@example.com"
    }
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        result = await auth_service.verify_token("valid-token")
        
        assert result["id"] == "user123"
        assert result["email"] == "test@example.com"

@pytest.mark.asyncio
async def test_credential_service_get_openai_token():
    """Test getting OpenAI credentials"""
    credential_service = CredentialService()
    
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "sk-test-token"
    }
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        token = await credential_service.get_openai_token("user123")
        
        assert token == "sk-test-token"
