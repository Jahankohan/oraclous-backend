import logging

from fastapi import Request
from repositories.workflow_repository import WorkflowRepository

logger = logging.getLogger(__name__)


async def get_repository(request: Request) -> WorkflowRepository:
    try:
        # Access the repository from app.state
        return request.app.state.repository
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