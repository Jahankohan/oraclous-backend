import contextvars

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, AsyncGenerator
from app.core.database import get_db
from app.services.auth_service import auth_service
from app.services.rebac_service import rebac_service

security = HTTPBearer()

# Stores the resolved principal dict for the current request (set in get_current_user).
# Enables verify_graph_access to route SA vs user permission checks without
# changing the call signature used by the 40+ existing endpoint callers.
_current_principal: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "current_principal", default={}
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Dependency to get current authenticated user or service account principal."""
    token = credentials.credentials
    user = await auth_service.verify_token(token)
    # Store principal in contextvar so verify_graph_access can branch on principal_type
    _current_principal.set(user)
    return user


async def get_current_user_id(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> str:
    """Dependency to get current principal ID (user_id or service_account_id)."""
    return str(current_user["id"])


# Re-export database dependency
async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    async for session in get_db():
        yield session


async def verify_graph_access(
    graph_id: str,
    required_level: str,
    user_id: str,
) -> str:
    """
    Verify the current principal has at least required_level access to graph_id.

    Routes to service-account ACL check or user ReBAC check based on principal_type
    stored in the _current_principal contextvar (set by get_current_user).

    Always returns HTTP 403 on denial — never 404 — to prevent graph_id
    enumeration attacks (ORA-39 spec, Section 3.5).

    Returns graph_id if authorized.
    """
    from app.core.neo4j_client import neo4j_client

    if not neo4j_client.async_driver:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j connection not available",
        )

    principal = _current_principal.get()
    principal_type = principal.get("principal_type", "user")

    if principal_type == "service_account":
        from app.services.service_account_service import service_account_service
        tenant_id = principal.get("tenant_id", "")
        authorized = await service_account_service.check_sa_graph_permission(
            driver=neo4j_client.async_driver,
            sa_id=user_id,
            tenant_id=tenant_id,
            graph_id=graph_id,
            required_level=required_level,
        )
    else:
        authorized = await rebac_service.check_graph_permission(
            driver=neo4j_client.async_driver,
            user_id=user_id,
            graph_id=graph_id,
            required_level=required_level,
        )

    if not authorized:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return graph_id
