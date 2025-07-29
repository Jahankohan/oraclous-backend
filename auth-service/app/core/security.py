import time
from jose import jwt
from app.core.config import settings

def sign_state(payload: dict, expires_in: int = 600):
    payload["exp"] = int(time.time()) + expires_in
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

def decode_state(token: str):
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
