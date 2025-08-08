import logging

from fastapi import Request
from app.repositories.credential_repository import CredentialRepository

logger = logging.getLogger(__name__)

async def get_credential_repository(request: Request) -> CredentialRepository:
    try:
        return request.app.state.credential_repository
    except AttributeError:
        raise ValueError("CredentialRepository not initialized in app.state")
