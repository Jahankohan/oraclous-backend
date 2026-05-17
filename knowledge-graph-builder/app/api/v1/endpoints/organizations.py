"""Organization API endpoints (TASK-201).

Makes :Organization a first-class entity with CRUD over the PostgreSQL
``organizations`` table plus a synced Neo4j ``:Organization`` node.

Routes:
  POST   /organizations            — create an org (current user becomes owner)
  GET    /organizations            — list orgs owned by the current user
  GET    /organizations/{org_id}   — get one org (owner-only)
  PATCH  /organizations/{org_id}   — update one org (owner-only)

Access control for TASK-201 is owner-only: the current user must equal the
org's ``owner_user_id``. The not-owner case returns 404 (not 403) so org
existence is never leaked. Membership-based access is a later task (TASK-203).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user_id, get_database
from app.core.dependencies import get_neo4j_async_driver
from app.models.organization import Organization
from app.schemas.organization_schemas import (
    OrganizationCreate,
    OrganizationResponse,
    OrganizationUpdate,
)
from app.services import organization_service

router = APIRouter()


def _serialize(org: Organization) -> OrganizationResponse:
    """Map a SQL Organization row to the API response (ISO-8601 datetimes)."""
    return OrganizationResponse(
        id=str(org.id),
        name=org.name,
        description=org.description or "",
        owner_user_id=str(org.owner_user_id),
        settings=org.settings or {},
        status=org.status,
        created_at=org.created_at.isoformat() if org.created_at else "",
        updated_at=org.updated_at.isoformat() if org.updated_at else "",
    )


@router.post(
    "/organizations",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_organization(
    body: OrganizationCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> OrganizationResponse:
    """Create an organization. The current user becomes its owner."""
    org = await organization_service.create_organization(
        db,
        driver,
        name=body.name,
        description=body.description,
        settings=body.settings,
        owner_user_id=user_id,
    )
    return _serialize(org)


@router.get(
    "/organizations",
    response_model=list[OrganizationResponse],
)
async def list_organizations(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> list[OrganizationResponse]:
    """List all organizations owned by the current user."""
    orgs = await organization_service.list_organizations(db, user_id)
    return [_serialize(o) for o in orgs]


@router.get(
    "/organizations/{org_id}",
    response_model=OrganizationResponse,
)
async def get_organization(
    org_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
) -> OrganizationResponse:
    """Get a single organization. Owner-only — 404 if missing or not owned."""
    org = await organization_service.get_organization(db, org_id)
    if org is None or str(org.owner_user_id) != user_id:
        # Never leak existence to non-owners.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    return _serialize(org)


@router.patch(
    "/organizations/{org_id}",
    response_model=OrganizationResponse,
)
async def update_organization(
    org_id: str,
    body: OrganizationUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database),
    driver: AsyncDriver = Depends(get_neo4j_async_driver),
) -> OrganizationResponse:
    """Update an organization. Owner-only — 404 if missing or not owned."""
    org = await organization_service.get_organization(db, org_id)
    if org is None or str(org.owner_user_id) != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )

    updated = await organization_service.update_organization(
        db,
        driver,
        org_id,
        name=body.name,
        description=body.description,
        settings=body.settings,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found"
        )
    return _serialize(updated)
