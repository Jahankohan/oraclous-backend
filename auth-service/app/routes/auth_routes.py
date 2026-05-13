import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app.core.dependencies import (
    get_service_account_repository,
    get_user_repository,
    verify_internal_service,
)
from app.core.jwt_handler import create_service_account_token, verify_access_token
from app.core.rate_limiter import enforce_key_prefix_rate_limit, limiter
from app.schema import auth_schemas
from app.services.auth_service import AuthService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
router = APIRouter()


@router.post("/register/", response_model=auth_schemas.Token, status_code=201)
@limiter.limit("5/minute")
async def register_user(user: auth_schemas.UserCreateWithEmail, request: Request):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    token_schema = await auth_service.create_user(user=user)
    return token_schema


@router.post("/login/", response_model=auth_schemas.Token)
@limiter.limit("10/minute")
async def login_user(form_data: auth_schemas.UserLogin, request: Request):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)
    token_schema = await auth_service.authenticate_user(
        email=form_data.email, password=form_data.password
    )
    if not token_schema:
        raise HTTPException(
            status_code=401, detail="Incorrect email/username or password"
        )
    return token_schema


@router.post("/refresh/", response_model=auth_schemas.Token)
@limiter.limit("20/minute")
async def refresh_token(
    refresh_token: auth_schemas.RefreshTokenRequest, request: Request
):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    token_schema = await auth_service.verify_refresh_token(refresh_token.refresh_token)
    if not token_schema:
        raise HTTPException(
            status_code=400, detail="Refresh Token is invalid or expired"
        )
    return token_schema


@router.post("/verify-email/")
async def verify_email(
    email_verification_request: auth_schemas.EmailVerificationRequest,
    request: Request,
    token: str = Depends(oauth2_scheme),
):
    print("Received token:", token)
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    code = email_verification_request.code
    print("Verification code:", code)
    return await auth_service.verify_user_email(token, code)


@router.post("/resend-email-verification/")
async def resend_email_verification(
    request: Request, token: str = Depends(oauth2_scheme)
):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    return await auth_service.resend_verification_email(token=token)


@router.post("/forgot-password/")
@limiter.limit("5/minute")
async def forgot_password(
    forget_password_req: auth_schemas.ForgotPasswordRequest, request: Request
):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    return await auth_service.send_password_reset_email(email=forget_password_req.email)


@router.post("/reset-password/")
@limiter.limit("5/minute")
async def reset_password(
    token: str, reset_password_req: auth_schemas.ResetPasswordRequest, request: Request
):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    return await auth_service.reset_password(
        token=token, new_password=reset_password_req.new_password
    )


@router.post("/change-password/")
async def change_password(
    change_password_req: auth_schemas.ChangePasswordRequest,
    request: Request,
    token: str = Depends(oauth2_scheme),
):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    payload = await auth_service.verify_access_token(token)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    db_user = await repository.get_user_by_id(user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return await auth_service.change_password(
        user=db_user, new_password=change_password_req.new_password
    )


@router.get("/validate")
async def validate_token(request: Request, token: str = Depends(oauth2_scheme)):
    repository = await get_user_repository(request)
    auth_service = AuthService(repository)

    payload = await auth_service.verify_access_token(token)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    db_user = await repository.get_user_by_id(user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return payload


@router.get("/me")
async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    """Get current user (or service account) info — used by all downstream services."""
    payload = verify_access_token(token)
    principal_type = payload.get("principal_type", "user")
    sub = payload.get("sub")

    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    if principal_type == "service_account":
        # Verify the SA still has an active key (revocation check)
        sa_repository = await get_service_account_repository(request)
        is_active = await sa_repository.has_active_key(sub)
        if not is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Service account has been revoked",
            )
        # Update last_used_at asynchronously (non-blocking per spec)
        asyncio.create_task(sa_repository.update_last_used(sub))
        return {
            "id": sub,
            "principal_type": "service_account",
            "tenant_id": payload.get("tenant_id"),
            "home_graph_id": payload.get("home_graph_id"),
        }

    # Standard user path
    repository = await get_user_repository(request)

    db_user = await repository.get_user_by_id(sub)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return {
        "id": str(db_user.id),
        "email": db_user.email,
        "first_name": db_user.first_name,
        "last_name": db_user.last_name,
        "is_verified": db_user.is_email_verified,
        "principal_type": "user",
        "home_graph_id": db_user.home_graph_id,
    }


# ── Service Account Endpoints ──────────────────────────────────────────────


class ServiceTokenRequest(BaseModel):
    api_key: str


class ServiceTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    principal_type: str = "service_account"


@router.post("/service-token", response_model=ServiceTokenResponse)
@limiter.limit("10/minute")
async def exchange_service_token(
    request: Request,
    body: ServiceTokenRequest,
    _prefix_limit: None = Depends(enforce_key_prefix_rate_limit),
):
    """Exchange an osk_ API key for a short-lived JWT (15 min).

    The issued JWT carries principal_type=service_account so all downstream
    services can branch on it without additional DB lookups.
    """
    sa_repository = await get_service_account_repository(request)
    sa_id = await sa_repository.validate_key(body.api_key)
    if not sa_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
        )

    # Fetch SA metadata from Neo4j via KGB is not possible here (auth-service
    # has no Neo4j access). tenant_id and home_graph_id are stored in a minimal
    # lookup. For now we embed these as opaque claims — KGB populates them at
    # SA creation time by calling the internal key creation endpoint which stores
    # the metadata alongside the key.
    #
    # Retrieve stored metadata from the key record via SA lookup
    from sqlalchemy.future import select

    from app.models.service_account_model import AgentServiceAccountKey

    async with sa_repository.Session() as session:
        result = await session.execute(
            select(AgentServiceAccountKey)
            .where(
                AgentServiceAccountKey.service_account_id == sa_id,
                AgentServiceAccountKey.status == "active",
            )
            .limit(1)
        )
        key_record = result.scalars().first()

    if not key_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
        )

    tenant_id = getattr(key_record, "tenant_id", None) or ""
    home_graph_id = getattr(key_record, "home_graph_id", None) or ""

    token, expires_in = create_service_account_token(sa_id, tenant_id, home_graph_id)
    return ServiceTokenResponse(access_token=token, expires_in=expires_in)


# ── Internal endpoints (service-to-service, requires X-Internal-Key header) ─


class CreateKeyRequest(BaseModel):
    service_account_id: str
    created_by_user_id: str
    tenant_id: str
    home_graph_id: str
    expires_at: Optional[str] = None


class CreateKeyResponse(BaseModel):
    key_id: str
    api_key: str  # raw key — shown only once
    key_prefix: str


class RevokeKeysResponse(BaseModel):
    revoked_count: int


@router.post("/internal/service-account-keys", response_model=CreateKeyResponse)
async def internal_create_sa_key(
    body: CreateKeyRequest,
    request: Request,
    _: bool = Depends(verify_internal_service),
):
    """Internal endpoint: create a new API key for a service account.

    Called by knowledge-graph-builder during SA creation and key rotation.
    Returns the raw API key ONCE — never stored in plaintext.
    """
    from datetime import datetime

    expires_at = None
    if body.expires_at:
        expires_at = datetime.fromisoformat(body.expires_at.replace("Z", "+00:00"))

    sa_repository = await get_service_account_repository(request)
    raw_key, record = await sa_repository.create_key(
        service_account_id=body.service_account_id,
        created_by_user_id=body.created_by_user_id,
        expires_at=expires_at,
    )

    # Persist tenant_id and home_graph_id on the key record for token exchange
    from sqlalchemy import update as sa_update

    from app.models.service_account_model import AgentServiceAccountKey

    async with sa_repository.Session() as session:
        await session.execute(
            sa_update(AgentServiceAccountKey)
            .where(AgentServiceAccountKey.key_id == record.key_id)
            .values(
                tenant_id=body.tenant_id,
                home_graph_id=body.home_graph_id,
            )
        )
        await session.commit()

    return CreateKeyResponse(
        key_id=record.key_id,
        api_key=raw_key,
        key_prefix=record.key_prefix,
    )


class SetHomeGraphRequest(BaseModel):
    home_graph_id: str


class SetHomeGraphResponse(BaseModel):
    user_id: str
    home_graph_id: str


@router.put("/internal/users/{user_id}/home-graph", response_model=SetHomeGraphResponse)
async def internal_set_home_graph(
    user_id: str,
    body: SetHomeGraphRequest,
    request: Request,
    _: bool = Depends(verify_internal_service),
):
    """Internal: bind a user's default workspace.

    Called by knowledge-graph-builder during onboarding bootstrap after a
    workspace (graph) has been created for the user.
    """
    repository = await get_user_repository(request)
    user = await repository.set_home_graph_id(user_id, body.home_graph_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return SetHomeGraphResponse(user_id=str(user.id), home_graph_id=user.home_graph_id)


@router.post("/internal/users/{user_id}/mint-token", response_model=auth_schemas.Token)
async def internal_mint_user_token(
    user_id: str,
    request: Request,
    _: bool = Depends(verify_internal_service),
):
    """Internal: mint a fresh access+refresh pair for an existing user.

    Used by knowledge-graph-builder right after binding a home_graph_id so the
    caller can hand back a token that already carries the workspace claim
    (avoiding a second /refresh/ round-trip from the client).
    """
    repository = await get_user_repository(request)
    db_user = await repository.get_user_by_id(user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    auth_service = AuthService(repository)
    access_token, expires_in = await auth_service.create_access_token(
        data={
            "sub": str(db_user.id),
            "email": db_user.email,
            "is_superuser": db_user.is_superuser,
            "home_graph_id": db_user.home_graph_id,
        }
    )
    refresh_token = await auth_service.create_refresh_token(
        data={
            "sub": str(db_user.id),
            "email": db_user.email,
            "is_superuser": db_user.is_superuser,
            "home_graph_id": db_user.home_graph_id,
        }
    )
    return auth_schemas.Token(
        email=db_user.email,
        is_superuser=db_user.is_superuser,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        token_type="bearer",
    )


@router.delete(
    "/internal/service-account-keys/{service_account_id}",
    response_model=RevokeKeysResponse,
)
async def internal_revoke_sa_keys(
    service_account_id: str,
    request: Request,
    _: bool = Depends(verify_internal_service),
):
    """Internal endpoint: revoke all active keys for a service account.

    Called by knowledge-graph-builder during SA deletion or key rotation.
    """
    sa_repository = await get_service_account_repository(request)
    count = await sa_repository.revoke_keys_for_sa(service_account_id)
    return RevokeKeysResponse(revoked_count=count)
