import logging

from fastapi import Request
from app.repositories.credential_repository import CredentialRepository
from app.core.config import settings
from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

async def get_credential_repository(request: Request) -> CredentialRepository:
    try:
        return request.app.state.credential_repository
    except AttributeError:
        raise ValueError("CredentialRepository not initialized in app.state")


async def verify_internal_service(x_internal_service_key: str = Header(...)):
    """Verify that the request comes from an internal service."""
    expected_key = settings.INTERNAL_SERVICE_KEY
    logger.info(f"Verifying internal service with key: {x_internal_service_key}")
    if not expected_key or x_internal_service_key != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized internal service call")
    return True