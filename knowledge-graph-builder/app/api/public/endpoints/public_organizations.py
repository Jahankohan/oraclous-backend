"""Public, unauthenticated organization lookups (subdomain tenant routing).

These endpoints back the frontend's pre-login decisions on a company
subdomain (``<slug>.oraclous.com``) and on the invitation screen. They are
unauthenticated by design and deliberately return only minimal, non-sensitive
fields — no owner, members, settings, or counts.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_database
from app.schemas.organization_schemas import (
    PublicInvitationResponse,
    PublicOrganizationResponse,
)
from app.services import org_invitation_service, organization_service

router = APIRouter()


@router.get(
    "/organizations/by-slug/{slug}",
    response_model=PublicOrganizationResponse,
    summary="Resolve an organization subdomain slug to minimal public info",
)
async def get_organization_by_slug(
    slug: str,
    db: AsyncSession = Depends(get_database),
) -> PublicOrganizationResponse:
    org = await organization_service.get_organization_by_slug(db, slug)
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )
    return PublicOrganizationResponse(
        id=str(org.id),
        name=org.name,
        slug=org.slug,
        status=org.status,
        logo_url=org.logo_url,
    )


@router.get(
    "/invitations/{token}",
    response_model=PublicInvitationResponse,
    summary="Peek at an invitation's organization (name + logo) before accepting",
)
async def peek_invitation(
    token: str,
    db: AsyncSession = Depends(get_database),
) -> PublicInvitationResponse:
    invitation = await org_invitation_service.get_invitation_by_token(db, token)
    if invitation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found",
        )
    org = await organization_service.get_organization(db, str(invitation.org_id))
    if org is None:
        # Invitation references an org that no longer exists — treat as gone.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found",
        )
    return PublicInvitationResponse(
        org_name=org.name,
        org_logo_url=org.logo_url,
        invited_email=invitation.email,
        status=invitation.status,
    )
