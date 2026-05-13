"""Unit tests: home_graph_id claim flows through access + refresh token mint.

Covers the fresh-user onboarding contract:
- When the user has no home_graph_id yet, the issued token must NOT carry the
  claim (avoids a 'null' claim leaking into the JWT).
- When the user has a home_graph_id, both access and refresh tokens carry it.
- Existing claims (sub, email, is_superuser, exp, type) stay intact.
"""

import uuid

from jose import jwt

from app.core.config import settings
from app.core.jwt_handler import create_access_token, create_refresh_token


def _decode(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])


def test_access_token_omits_home_graph_id_when_absent():
    user_id = str(uuid.uuid4())
    token, _ = create_access_token(
        {"sub": user_id, "email": "a@b.com", "is_superuser": False}
    )
    payload = _decode(token)
    assert "home_graph_id" not in payload
    assert payload["sub"] == user_id
    assert payload["type"] == "access"


def test_access_token_omits_home_graph_id_when_none():
    """A user who has not been bootstrapped yet has home_graph_id=None on the
    DB record. The mint must not put a 'null' claim into the JWT.
    """
    user_id = str(uuid.uuid4())
    token, _ = create_access_token(
        {
            "sub": user_id,
            "email": "a@b.com",
            "is_superuser": False,
            "home_graph_id": None,
        }
    )
    payload = _decode(token)
    assert "home_graph_id" not in payload


def test_access_token_includes_home_graph_id_when_set():
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    token, _ = create_access_token(
        {
            "sub": user_id,
            "email": "a@b.com",
            "is_superuser": False,
            "home_graph_id": graph_id,
        }
    )
    payload = _decode(token)
    assert payload["home_graph_id"] == graph_id
    assert payload["sub"] == user_id
    assert payload["email"] == "a@b.com"


def test_refresh_token_includes_home_graph_id_when_set():
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    token = create_refresh_token(
        {
            "sub": user_id,
            "email": "a@b.com",
            "is_superuser": False,
            "home_graph_id": graph_id,
        }
    )
    payload = _decode(token)
    assert payload["home_graph_id"] == graph_id
    assert payload["type"] == "refresh"


def test_refresh_token_omits_home_graph_id_when_absent():
    user_id = str(uuid.uuid4())
    token = create_refresh_token(
        {"sub": user_id, "email": "a@b.com", "is_superuser": False}
    )
    payload = _decode(token)
    assert "home_graph_id" not in payload
