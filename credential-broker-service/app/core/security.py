import os, base64, json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from datetime import datetime, timedelta
from typing import Union, Dict
from app.core.config import settings

def encrypt_secret(secret: Union[str, Dict]) -> str:
    """Encrypt a secret (string or dict) and return hex-encoded encrypted data."""
    key = base64.b64decode(settings.ENCRYPTION_KEY)
    aesgcm = AESGCM(key)

    # Convert dict to JSON string if needed
    if isinstance(secret, dict):
        secret_str = json.dumps(secret)
    else:
        secret_str = secret

    nonce = os.urandom(12)
    encrypted = aesgcm.encrypt(nonce, secret_str.encode(), None)
    return (nonce + encrypted).hex()

def decrypt_secret(encrypted_hex: str) -> Union[str, Dict]:
    """Decrypt hex-encoded encrypted data and return original format."""
    data = bytes.fromhex(encrypted_hex)
    key = base64.b64decode(settings.ENCRYPTION_KEY)
    aesgcm = AESGCM(key)
    
    nonce, ciphertext = data[:12], data[12:]
    decrypted_str = aesgcm.decrypt(nonce, ciphertext, None).decode()
    
    # Try to parse as JSON, return as dict if successful, otherwise return string
    try:
        return json.loads(decrypted_str)
    except json.JSONDecodeError:
        return decrypted_str
