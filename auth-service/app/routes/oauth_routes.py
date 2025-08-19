from fastapi import APIRouter, Request, HTTPException, Depends, Query
from fastapi.responses import RedirectResponse, JSONResponse
from typing import List, Optional
from datetime import datetime

from app.core.dependencies import get_token_repository, get_user_repository, verify_internal_service
from app.core.jwt_handler import decode_state, create_access_token, create_refresh_token
from urllib.parse import urlencode
from app.schema.oauth_schemas import TokenRefreshRequest, TokenRefreshResponse, ScopeValidationRequest, RuntimeTokenResponse, ScopeValidationResponse, UserTokensResponse, EnsureAccessResponse
from app.services.oauth_service import OAuthService
from pydantic import BaseModel

router = APIRouter()

# Existing endpoints remain the same...
@router.get("/oauth/{provider}/login")
async def login(provider: str, request: Request, state: str = "/"):
    """Redirect to OAuth login page for the provider."""
    repository = await get_token_repository(request)
    oauth_service = OAuthService(repository)
    login_url = await oauth_service.build_login_url(provider, state)
    return RedirectResponse(login_url)


@router.get("/oauth/{provider}/callback")
async def callback(provider: str, request: Request):
    """Handle OAuth callback, exchange code, save token, redirect user."""
    code = request.query_params.get("code")
    state_jwt = request.query_params.get("state")

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    try:
        state_data = decode_state(state_jwt)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or expired state")

    redirect_path = state_data.get("state", "/")

    token_repository = await get_token_repository(request)
    user_repository = await get_user_repository(request)

    oauth_service = OAuthService(token_repository)

    # 1. Exchange code for token
    token_data = await oauth_service.exchange_token(provider, code)

    # 2. Fetch user profile from provider
    profile = await oauth_service.fetch_user_profile(provider, token_data["access_token"])
    email = profile["email"]

    # 3. Check if user exists
    user = await user_repository.get_user_by_email(email)
    if not user:
        first_name = profile.get("first_name", "")
        last_name = profile.get("last_name", "")
        picture = profile.get("picture", "")
        user = await user_repository.create_user(email=email, first_name=first_name, last_name=last_name, profile_picture=picture)

    await token_repository.save_token(
        user_id=user.id,
        provider=provider,
        access_token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        scopes=token_data.get("scopes", []),
        expires_at=token_data.get("expires_at"),
    )

    jwt_token, _ = create_access_token({"sub": email, "is_superuser": False})
    refresh_token = create_refresh_token({"sub": email, "is_superuser": False})
    print("Access token:", jwt_token)
    params = {
        "access_token": jwt_token,
        "refresh_token": refresh_token,
        "email": email,
        "state": redirect_path
    }
    
    # Build the frontend callback URL
    frontend_base_url = "http://localhost:8080/oauth/{}/callback".format(provider)
    query_string = urlencode(params)
    response_url = f"{frontend_base_url}?{query_string}"
    
    return RedirectResponse(url=response_url)

@router.get("/oauth/login-url")
async def get_login_url(
    provider: str = Query(...),
    state: str = Query("/"),
    required_scopes: Optional[List[str]] = Query(None),
    request: Request = None,
    _: bool = Depends(verify_internal_service)
):
    """Return OAuth login URL for a provider and user (internal service only)."""
    repository = await get_token_repository(request)
    oauth_service = OAuthService(repository)
    login_url = await oauth_service.build_login_url(provider, state, required_scopes)
    return {"login_url": login_url}

@router.post("/oauth/validate-scopes", response_model=ScopeValidationResponse)
async def validate_scopes(
    request_data: ScopeValidationRequest, 
    request: Request,
    _: bool = Depends(verify_internal_service)
):
    """Validate if user has required scopes for a provider (internal service only)."""
    token_repository = await get_token_repository(request)
    oauth_service = OAuthService(token_repository)
    
    result = await oauth_service.ensure_access(
        user_id=request_data.user_id,
        provider=request_data.provider,
        required_scopes=request_data.required_scopes,
        redirect_state="/oauth-callback"  # Default state for re-auth
    )
    
    if result["action"] == "ok":
        return ScopeValidationResponse(
            valid=True,
            missing_scopes=[],
            current_scopes=result["current_scopes"],
            token_expired=False,
            needs_reauth=False
        )
    else:
        return ScopeValidationResponse(
            valid=False,
            missing_scopes=result["missing_scopes"],
            current_scopes=result["current_scopes"],
            token_expired=True,
            needs_reauth=True,
            login_url=result["login_url"]
        )

@router.post("/oauth/refresh-if-needed", response_model=TokenRefreshResponse)
async def refresh_if_needed(
    request_data: TokenRefreshRequest,
    request: Request,
    _: bool = Depends(verify_internal_service)
):
    """Refresh token if expired (internal service only)."""
    token_repository = await get_token_repository(request)
    oauth_service = OAuthService(token_repository)
    
    try:
        token_obj = await token_repository.get_token(request_data.user_id, request_data.provider)

        if not token_obj:
            login_url = await oauth_service.build_login_url(request_data.provider, state=request_data.state)
            return TokenRefreshResponse(
                success=False,
                error="Token not found",
                login_url=login_url
            )
        
        # Check if token is expired
        if token_obj.expires_at and datetime.utcnow() > token_obj.expires_at:
            if token_obj.refresh_token:
                # Refresh the token
                refreshed = await oauth_service.refresh_token(
                    request_data.user_id, 
                    request_data.provider, 
                    token_obj.refresh_token
                )
                return TokenRefreshResponse(
                    success=True,
                    access_token=refreshed["access_token"],
                    expires_at=refreshed["expires_at"]
                )
            else:
                login_url = await oauth_service.build_login_url(request_data.provider, state=request_data.state)
                return TokenRefreshResponse(
                    success=False,
                    error="Token expired and no refresh token available",
                    login_url=login_url
                )
        else:
            # Token is still valid
            return TokenRefreshResponse(
                success=True,
                access_token=token_obj.access_token,
                expires_at=token_obj.expires_at
            )
            
    except Exception as e:
        login_url = await oauth_service.build_login_url(request_data.provider, state=request_data.state)
        return TokenRefreshResponse(
            success=False,
            error=f"Token refresh failed: {str(e)}",
            login_url=login_url
        )

@router.get("/oauth/user-tokens", response_model=UserTokensResponse)
async def get_user_tokens(
    request: Request,
    user_id: str = Query(...),
    _: bool = Depends(verify_internal_service)
):
    """Get all OAuth tokens for a user (internal service only)."""
    token_repository = await get_token_repository(request)
    
    tokens = await token_repository.list_tokens(user_id)
    
    providers = []
    for token in tokens:
        providers.append({
            "provider": token.provider,
            "scopes": token.scopes or [],
            "expires_at": token.expires_at,
            "has_refresh_token": bool(token.refresh_token)
        })
    
    return UserTokensResponse(
        user_id=user_id,
        providers=providers
    )


@router.post("/oauth/{provider}/ensure-access")
async def ensure_access(provider: str, request: Request):
    """Check if user has required scopes or re-authenticate if needed."""
    data = await request.json()
    user_id = data["user_id"]
    required_scopes = data.get("required_scopes", [])
    state = data.get("state", "/")

    repository = await get_token_repository(request)
    oauth_service = OAuthService(repository)
    result = await oauth_service.ensure_access(user_id, provider, required_scopes, state)
    return result

@router.get("/oauth/runtime-tokens")
async def get_runtime_token(
    request: Request,
    user_id: str = Query(...),
    provider: str = Query(...),
    _=Depends(verify_internal_service)
):
    """Retrieve runtime OAuth token for a specific user and provider (internal use only)."""
    token_repository = await get_token_repository(request)
    token_obj = await token_repository.get_token(user_id, provider)

    if not token_obj:
        raise HTTPException(status_code=404, detail="Token not found")
    
    # Check if token is expired and refresh if needed
    if token_obj.expires_at and token_obj.expires_at <= datetime.utcnow():
        if token_obj.refresh_token:
            oauth_service = OAuthService(token_repository)
            try:
                refreshed = await oauth_service.refresh_token(user_id, provider, token_obj.refresh_token)
                runtime_token_response = RuntimeTokenResponse(
                     user_id=user_id,
                     provider=provider,
                     access_token=refreshed["access_token"],
                     expires_at=refreshed["expires_at"],
                     scopes=token_obj.scopes,
                     refresh_token=token_obj.refresh_token
                )
                return runtime_token_response
            except Exception as e:
                raise HTTPException(status_code=401, detail=f"Token refresh failed: {str(e)}")
        else:
            raise HTTPException(status_code=401, detail="Token expired and no refresh token available")
    runtime_token_response = RuntimeTokenResponse(
        user_id=user_id,
        provider=provider,
        access_token=token_obj.access_token,
        expires_at=token_obj.expires_at,
        scopes=token_obj.scopes,
        refresh_token=token_obj.refresh_token
    )
    return runtime_token_response
