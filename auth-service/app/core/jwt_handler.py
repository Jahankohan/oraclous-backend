import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.config import settings
from app.schema.auth_schemas import TokenData
from fastapi import HTTPException, status
from jose import JWTError, jwt

# Service account JWTs are short-lived (15 minutes) per security spec
_SA_TOKEN_EXPIRE_MINUTES = 15


credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

outdated_token_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Outdated token format — please re-authenticate",
    headers={"WWW-Authenticate": "Bearer"},
)


def _is_email(value: str) -> bool:
    """Detect legacy tokens where sub was set to email instead of user_id."""
    return "@" in value


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = {k: v for k, v in data.items() if v is not None}
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=int(settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
    # sub MUST be user_id (UUID), email stored as separate claim
    claims = {
        "exp": expire,
        "type": "access",
        "sub": data.get("sub"),  # caller must set sub=str(user_id)
        "email": data.get("email"),
        "is_superuser": data.get("is_superuser"),
    }
    to_encode.update(claims)
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt, int(settings.ACCESS_TOKEN_EXPIRE_MINUTES) * 60


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = {k: v for k, v in data.items() if v is not None}
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=int(settings.REFRESH_TOKEN_EXPIRE_DAYS)
        )
    # sub MUST be user_id (UUID), email stored as separate claim
    claims = {
        "exp": expire,
        "type": "refresh",
        "sub": data.get("sub"),  # caller must set sub=str(user_id)
        "email": data.get("email"),
        "is_superuser": data.get("is_superuser"),
    }
    to_encode.update(claims)
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def verify_access_token(token: str):
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        # Reject legacy tokens that still have email in sub
        if _is_email(user_id):
            raise outdated_token_exception
        return payload
    except HTTPException:
        raise
    except JWTError:
        raise credentials_exception


def verify_refresh_token(token: str):
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        if _is_email(user_id):
            raise outdated_token_exception
        return TokenData(
            user_id=user_id, is_superuser=payload.get("is_superuser", False)
        )
    except HTTPException:
        raise
    except JWTError:
        raise credentials_exception


def create_service_account_token(
    sa_id: str, tenant_id: str, home_graph_id: str
) -> tuple[str, int]:
    """Issue a short-lived JWT (15 min) for an AgentServiceAccount principal.

    JWT payload extends ORA-62 structure with principal_type and home_graph_id.
    Returns (access_token, expires_in_seconds).
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=_SA_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": sa_id,
        "tenant_id": tenant_id,
        "principal_type": "service_account",
        "home_graph_id": home_graph_id,
        "iat": datetime.now(timezone.utc),
        "exp": expire,
        "jti": str(uuid.uuid4()),
    }
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token, _SA_TOKEN_EXPIRE_MINUTES * 60


def sign_state(payload: dict, expires_in: int = 600):
    payload["exp"] = int(time.time()) + expires_in
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_state(token: str):
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
