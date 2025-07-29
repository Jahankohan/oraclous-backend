from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse

from app.core.dependencies import get_repository
from app.core.security import decode_state
from app.services.oauth_service import OAuthService

router = APIRouter()

@router.get("/oauth/{provider}/login")
async def login(provider: str, request: Request, state: str = "/"):
    """Redirect to OAuth login page for the provider."""
    (repository, _) = await get_repository(request)
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

    (token_repository, user_repository) = await get_repository(request)
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
        user = await user_repository.create_user(email=email, first_name=first_name, last_name=last_name, profile_picture= picture)

    await token_repository.save_token(
        user_id=user.id,
        provider=provider,
        access_token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        scopes=token_data.get("scopes", []),
        expires_at=token_data.get("expires_at"),
    )

    return RedirectResponse(redirect_path)


@router.post("/oauth/{provider}/ensure-access")
async def ensure_access(provider: str, request: Request):
    """Check if user has required scopes or re-authenticate if needed."""
    data = await request.json()
    user_id = data["user_id"]
    required_scopes = data.get("required_scopes", [])
    state = data.get("state", "/")

    (repository, _) = await get_repository(request)
    oauth_service = OAuthService(repository)
    result = await oauth_service.ensure_access(user_id, provider, required_scopes, state)
    return JSONResponse(result)
