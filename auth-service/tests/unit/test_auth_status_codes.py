"""Unit tests for correct HTTP status codes on auth endpoints.

Covers three scenarios per ORA-226:
- POST /register/ success → 201 Created (verified via route decorator)
- POST /register/ duplicate email → 409 Conflict (verified in service layer)
- POST /login/ wrong password → 401 Unauthorized (verified in route layer)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status

# ---------------------------------------------------------------------------
# Route decorator: register must declare status_code=201
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_route_declares_201():
    """The /register/ route decorator must declare status_code=201."""
    from app.routes.auth_routes import router

    register_route = next(
        (r for r in router.routes if hasattr(r, "path") and r.path == "/register/"),
        None,
    )
    assert register_route is not None, "/register/ route not found"
    assert (
        register_route.status_code == 201
    ), f"Expected 201 but got {register_route.status_code}"


# ---------------------------------------------------------------------------
# AuthService.create_user — 409 on duplicate email
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_user_duplicate_raises_409():
    """Duplicate email registration must raise HTTPException with 409 Conflict."""
    from app.services.auth_service import AuthService

    mock_repo = AsyncMock()
    mock_repo.get_user_by_email.return_value = MagicMock()  # user already exists

    service = AuthService(mock_repo)

    with pytest.raises(HTTPException) as exc_info:
        from app.schema.auth_schemas import UserCreateWithEmail

        user = UserCreateWithEmail(email="dup@example.com", password="Pass123!")
        await service.create_user(user=user)

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "Email already registered"


# ---------------------------------------------------------------------------
# AuthService.authenticate_user — returns None on bad credentials
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_authenticate_user_wrong_password_returns_none():
    """authenticate_user returns None when password verification fails.

    The route then raises 401 — this validates the service contract.
    """
    from app.services.auth_service import AuthService

    mock_user = MagicMock()
    mock_repo = AsyncMock()
    mock_repo.get_user_by_email.return_value = mock_user

    service = AuthService(mock_repo)

    # Patch pwd_context.verify to simulate wrong password
    with patch("app.services.auth_service.pwd_context") as mock_pwd:
        mock_pwd.verify.return_value = False
        result = await service.authenticate_user(
            email="user@example.com", password="wrong_password"
        )

    assert result is None


# ---------------------------------------------------------------------------
# Route layer — login raises 401 on None return from service
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_login_route_raises_401_on_bad_credentials():
    """The login route must return HTTP 401 when authenticate_user returns None.

    Patches slowapi's _check_request_limit to bypass Redis so the route logic
    is tested in isolation without an external rate-limit backend.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    from app.core.rate_limiter import rate_limit_exceeded_handler

    test_limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
    app = FastAPI()
    app.state.limiter = test_limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    from app.routes.auth_routes import router

    app.include_router(router)

    mock_repo = AsyncMock()
    mock_service = AsyncMock()
    mock_service.authenticate_user.return_value = None

    # Patch _check_request_limit on the bound limiter to skip Redis entirely
    with patch(
        "app.routes.auth_routes.limiter._check_request_limit", return_value=None
    ), patch(
        "app.routes.auth_routes.get_user_repository", return_value=mock_repo
    ), patch(
        "app.routes.auth_routes.AuthService", return_value=mock_service
    ):
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post(
            "/login/",
            json={"email": "u@example.com", "password": "bad"},
        )

    assert r.status_code == 401
    assert r.json()["detail"] == "Incorrect email/username or password"
