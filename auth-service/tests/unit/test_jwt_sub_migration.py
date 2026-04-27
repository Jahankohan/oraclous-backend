"""
Unit tests for JWT sub claim migration — sub=user_id (UUID) instead of email.

Verifies:
- New tokens embed sub as UUID
- email stored as separate claim
- Legacy tokens (sub=email) are rejected with 401
- verify_access_token / verify_refresh_token enforce UUID sub
- TokenData uses user_id field
"""
import uuid
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from jose import jwt

# Patch settings before importing jwt_handler
FAKE_SECRET = "test-secret-key"
FAKE_ALGORITHM = "HS256"
FAKE_EXPIRE_MINUTES = "60"
FAKE_EXPIRE_DAYS = "7"

mock_settings = MagicMock()
mock_settings.JWT_SECRET = FAKE_SECRET
mock_settings.JWT_ALGORITHM = FAKE_ALGORITHM
mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = FAKE_EXPIRE_MINUTES
mock_settings.REFRESH_TOKEN_EXPIRE_DAYS = FAKE_EXPIRE_DAYS

with patch("app.core.jwt_handler.settings", mock_settings):
    from app.core.jwt_handler import (
        create_access_token,
        create_refresh_token,
        verify_access_token,
        verify_refresh_token,
        _is_email,
    )

from app.schema.auth_schemas import TokenData


# ---------------------------------------------------------------------------
# _is_email helper
# ---------------------------------------------------------------------------

def test_is_email_detects_email():
    assert _is_email("user@example.com") is True


def test_is_email_rejects_uuid():
    assert _is_email(str(uuid.uuid4())) is False


# ---------------------------------------------------------------------------
# create_access_token — sub must be UUID, email separate claim
# ---------------------------------------------------------------------------

def test_create_access_token_sub_is_user_id():
    user_id = str(uuid.uuid4())
    email = "user@example.com"

    with patch("app.core.jwt_handler.settings", mock_settings):
        token, expires_in = create_access_token({"sub": user_id, "email": email, "is_superuser": False})

    payload = jwt.decode(token, FAKE_SECRET, algorithms=[FAKE_ALGORITHM])
    assert payload["sub"] == user_id
    assert "@" not in payload["sub"]


def test_create_access_token_email_is_separate_claim():
    user_id = str(uuid.uuid4())
    email = "user@example.com"

    with patch("app.core.jwt_handler.settings", mock_settings):
        token, _ = create_access_token({"sub": user_id, "email": email, "is_superuser": False})

    payload = jwt.decode(token, FAKE_SECRET, algorithms=[FAKE_ALGORITHM])
    assert payload["email"] == email


def test_create_access_token_returns_expires_in():
    user_id = str(uuid.uuid4())

    with patch("app.core.jwt_handler.settings", mock_settings):
        _, expires_in = create_access_token({"sub": user_id, "email": "u@example.com", "is_superuser": False})

    assert expires_in == int(FAKE_EXPIRE_MINUTES) * 60


# ---------------------------------------------------------------------------
# create_refresh_token — same sub=UUID guarantee
# ---------------------------------------------------------------------------

def test_create_refresh_token_sub_is_user_id():
    user_id = str(uuid.uuid4())

    with patch("app.core.jwt_handler.settings", mock_settings):
        token = create_refresh_token({"sub": user_id, "email": "u@example.com", "is_superuser": False})

    payload = jwt.decode(token, FAKE_SECRET, algorithms=[FAKE_ALGORITHM])
    assert payload["sub"] == user_id


# ---------------------------------------------------------------------------
# verify_access_token — rejects legacy email-sub tokens
# ---------------------------------------------------------------------------

def _mint_legacy_token(email: str) -> str:
    """Mint a token with email in sub (legacy format)."""
    from datetime import datetime, timezone
    payload = {
        "sub": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    return jwt.encode(payload, FAKE_SECRET, algorithm=FAKE_ALGORITHM)


def _mint_valid_token(user_id: str) -> str:
    """Mint a token with UUID in sub (new format)."""
    from datetime import datetime, timezone
    payload = {
        "sub": user_id,
        "email": "user@example.com",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    return jwt.encode(payload, FAKE_SECRET, algorithm=FAKE_ALGORITHM)


def test_verify_access_token_accepts_uuid_sub():
    user_id = str(uuid.uuid4())
    token = _mint_valid_token(user_id)

    with patch("app.core.jwt_handler.settings", mock_settings):
        payload = verify_access_token(token)

    assert payload["sub"] == user_id


def test_verify_access_token_rejects_email_sub():
    token = _mint_legacy_token("user@example.com")

    with patch("app.core.jwt_handler.settings", mock_settings):
        with pytest.raises(HTTPException) as exc_info:
            verify_access_token(token)

    assert exc_info.value.status_code == 401
    assert "re-authenticate" in exc_info.value.detail.lower()


def test_verify_access_token_rejects_missing_sub():
    from datetime import datetime, timezone
    payload = {"exp": datetime.now(timezone.utc) + timedelta(hours=1)}
    token = jwt.encode(payload, FAKE_SECRET, algorithm=FAKE_ALGORITHM)

    with patch("app.core.jwt_handler.settings", mock_settings):
        with pytest.raises(HTTPException) as exc_info:
            verify_access_token(token)

    assert exc_info.value.status_code == 401


def test_verify_access_token_rejects_tampered_token():
    token = _mint_valid_token(str(uuid.uuid4())) + "tampered"

    with patch("app.core.jwt_handler.settings", mock_settings):
        with pytest.raises(HTTPException) as exc_info:
            verify_access_token(token)

    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# verify_refresh_token — returns TokenData with user_id field
# ---------------------------------------------------------------------------

def test_verify_refresh_token_returns_token_data_with_user_id():
    user_id = str(uuid.uuid4())

    with patch("app.core.jwt_handler.settings", mock_settings):
        refresh_token = create_refresh_token({"sub": user_id, "email": "u@example.com", "is_superuser": True})
        token_data = verify_refresh_token(refresh_token)

    assert isinstance(token_data, TokenData)
    assert token_data.user_id == user_id
    assert token_data.is_superuser is True


def test_verify_refresh_token_rejects_email_sub():
    token = _mint_legacy_token("user@example.com")

    with patch("app.core.jwt_handler.settings", mock_settings):
        with pytest.raises(HTTPException) as exc_info:
            verify_refresh_token(token)

    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# TokenData schema
# ---------------------------------------------------------------------------

def test_token_data_has_user_id_field():
    user_id = str(uuid.uuid4())
    td = TokenData(user_id=user_id, is_superuser=False)
    assert td.user_id == user_id


def test_token_data_has_no_email_field():
    td = TokenData(user_id=str(uuid.uuid4()))
    assert not hasattr(td, "email") or getattr(td, "email", None) is None
