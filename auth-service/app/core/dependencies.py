import logging

from fastapi import Request
from app.repositories.token_repository import TokenRepository
from app.repositories.user_repository import UserRepository
from typing import List

logger = logging.getLogger(__name__)


async def get_repository(request: Request):
    try:
        # Access the repository from app.state
        return (request.app.state.token_repository, request.app.state.user_repository)
    except AttributeError as e:
        # Handle the case where repository was not initialized in app.state
        logger.error(
            "Repository not initialized in app.state",
            extra={
                "error": str(e),
                "request_info": {
                    "method": request.method,
                    "url": str(request.url)
                },
                "class": "DifficultyAnalyticsService",
                "method": "get_repository"
            }
        )
        raise ValueError("Repository not found in application state")
    except Exception as e:
        # Handle any other potential issues
        logger.exception(
            "Unexpected error while retrieving the repository",
            extra={
                "error": str(e),
                "request_info": {
                    "method": request.method,
                    "url": str(request.url)
                },
                "class": "DifficultyAnalyticsService",
                "method": "get_repository"
            }
        )
        raise ValueError(f"Failed to retrieve repository: {str(e)}")