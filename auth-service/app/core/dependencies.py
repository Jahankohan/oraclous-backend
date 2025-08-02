import logging
import os

from fastapi import Request
from app.repositories.token_repository import TokenRepository
from app.repositories.user_repository import UserRepository
from app.core.config import settings
from fastapi import Header, HTTPException


logger = logging.getLogger(__name__)

async def get_token_repository(request: Request) -> TokenRepository:
    try:
        return request.app.state.token_repository
    except AttributeError:
        raise ValueError("TokenRepository not initialized in app.state")

async def get_user_repository(request: Request) -> UserRepository:
    try:
        return request.app.state.user_repository
    except AttributeError:
        raise ValueError("UserRepository not initialized in app.state")

async def verify_internal_service(x_internal_key: str = Header(...)):
    """Verify that the request comes from an internal service."""
    expected_key = settings.INTERNAL_SERVICE_KEY
    if not expected_key or x_internal_key != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized internal service call")
    return True
