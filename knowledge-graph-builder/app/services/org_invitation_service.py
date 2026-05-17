"""Organization member-invitation service.

An org owner or admin invites someone by email; the invitee accepts via an
opaque token and is registered as an org member. Invitations live in Postgres
(``org_invitations``); the Neo4j ``BELONGS_TO`` membership edge is created at
accept time by :func:`org_member_service.add_member`.

Statuses: ``pending`` → ``accepted`` | ``revoked`` | ``expired``.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID

from neo4j import AsyncDriver
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.organization import OrgInvitation
from app.services import org_member_service

logger = get_logger(__name__)

# ADR-021 §2 — the three org-level roles a member can be invited as.
_VALID_ORG_ROLES: frozenset[str] = frozenset({"owner", "admin", "member"})


def _is_expired(invitation: OrgInvitation) -> bool:
    """True when the invitation's ``expires_at`` is in the past."""
    expires = invitation.expires_at
    if expires is None:
        return False
    if expires.tzinfo is None:  # defensive — the column is timezone-aware
        expires = expires.replace(tzinfo=UTC)
    return datetime.now(UTC) >= expires


async def create_invitation(
    db: AsyncSession,
    *,
    org_id: str,
    email: str,
    org_role: str,
    invited_by_user_id: str,
) -> OrgInvitation:
    """Create a pending invitation and return it.

    Supersedes any still-pending invitation for the same (org, email) so only
    one invite per address is ever live. Raises ``ValueError`` on a bad
    ``org_role`` or a malformed email.
    """
    if org_role not in _VALID_ORG_ROLES:
        raise ValueError(
            f"Invalid org_role {org_role!r} — must be one of {sorted(_VALID_ORG_ROLES)}"
        )
    email = email.strip().lower()
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError(f"Invalid email address {email!r}")

    # Supersede any still-pending invite for the same address.
    await db.execute(
        update(OrgInvitation)
        .where(
            OrgInvitation.org_id == UUID(org_id),
            OrgInvitation.email == email,
            OrgInvitation.status == "pending",
        )
        .values(status="revoked")
    )

    invitation = OrgInvitation(
        org_id=UUID(org_id),
        email=email,
        org_role=org_role,
        token=secrets.token_urlsafe(32),
        status="pending",
        invited_by_user_id=UUID(invited_by_user_id),
        expires_at=datetime.now(UTC) + timedelta(hours=settings.INVITE_TTL_HOURS),
    )
    db.add(invitation)
    await db.commit()
    await db.refresh(invitation)
    logger.info("created invitation %s for %s -> org %s", invitation.id, email, org_id)
    return invitation


async def list_invitations(db: AsyncSession, org_id: str) -> list[OrgInvitation]:
    """Return every invitation for *org_id*, newest first."""
    result = await db.execute(
        select(OrgInvitation)
        .where(OrgInvitation.org_id == UUID(org_id))
        .order_by(OrgInvitation.created_at.desc())
    )
    return list(result.scalars().all())


async def revoke_invitation(db: AsyncSession, org_id: str, invitation_id: str) -> bool:
    """Revoke a pending invitation.

    Returns ``False`` when there is no *pending* invitation with that id in
    this org (already accepted/revoked/expired, or absent).
    """
    result = await db.execute(
        select(OrgInvitation).where(
            OrgInvitation.id == UUID(invitation_id),
            OrgInvitation.org_id == UUID(org_id),
        )
    )
    invitation = result.scalar_one_or_none()
    if invitation is None or invitation.status != "pending":
        return False
    invitation.status = "revoked"
    await db.commit()
    logger.info("revoked invitation %s", invitation_id)
    return True


async def get_invitation_by_token(db: AsyncSession, token: str) -> OrgInvitation | None:
    """Look up an invitation by its opaque token."""
    result = await db.execute(select(OrgInvitation).where(OrgInvitation.token == token))
    return result.scalar_one_or_none()


async def accept_invitation(
    db: AsyncSession,
    driver: AsyncDriver,
    *,
    token: str,
    accepting_user_id: str,
) -> dict:
    """Accept a pending invitation: register the accepting user as an org
    member and mark the invitation accepted.

    The token is the bearer credential — it was delivered to the invited
    email, so whoever holds a valid token and is authenticated joins the org
    under the invited email and role.

    Raises ``ValueError`` when the token is unknown, the invitation is not
    pending, or it has expired (an expired pending invite is flagged
    ``expired`` as a side effect).
    """
    invitation = await get_invitation_by_token(db, token)
    if invitation is None:
        raise ValueError("Invitation not found")
    if invitation.status != "pending":
        raise ValueError(f"Invitation is {invitation.status}, not pending")
    if _is_expired(invitation):
        invitation.status = "expired"
        await db.commit()
        raise ValueError("Invitation has expired")

    org_id = str(invitation.org_id)
    # Register the BELONGS_TO membership edge in Neo4j.
    await org_member_service.add_member(
        driver,
        org_id=org_id,
        user_id=accepting_user_id,
        org_role=invitation.org_role,
        email=invitation.email,
    )

    invitation.status = "accepted"
    invitation.accepted_by_user_id = UUID(accepting_user_id)
    invitation.accepted_at = datetime.now(UTC)
    await db.commit()
    logger.info("invitation %s accepted by user %s", invitation.id, accepting_user_id)
    return {
        "org_id": org_id,
        "org_role": invitation.org_role,
        "user_id": accepting_user_id,
    }
