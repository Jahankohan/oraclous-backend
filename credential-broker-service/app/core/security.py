import os, base64
from jose import jwt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from datetime import datetime, timedelta
from app.core.config import settings

def encrypt_secret(secret: str) -> str:
    key = base64.b64decode(settings.ENCRYPTION_KEY)
    aesgcm = AESGCM(key)

    nonce = os.urandom(12)
    encrypted = aesgcm.encrypt(nonce, secret.encode(), None)
    return (nonce + encrypted).hex()

def decrypt_secret(encrypted_hex: str) -> str:
    data = bytes.fromhex(encrypted_hex)
    key = base64.b64decode(settings.ENCRYPTION_KEY)
    aesgcm = AESGCM(key)
    
    nonce, ciphertext = data[:12], data[12:]
    return aesgcm.decrypt(nonce, ciphertext, None).decode()

def create_runtime_token(tenant_id: str) -> str:
    payload = {
        "sub": tenant_id,
        "exp": datetime.utcnow() + timedelta(minutes=int(settings.JWT_EXPIRY_MINUTES))
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

def verify_runtime_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
