"""HTTP client for the credential-broker-service (STORY-021).

Stores and retrieves LLM API keys via the credential-broker so plaintext keys
never touch Neo4j. Uses CredentialType.api_key with a fixed system sentinel
tool_id reserved for LLM configs.
"""

import uuid

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

# Sentinel tool_id for all LLM config credentials — never logged or returned to callers.
_LLM_CONFIG_TOOL_ID = uuid.UUID("00000000-0000-0000-0000-4c4c4d434647")


class CredentialBrokerError(Exception):
    """Raised when the credential-broker returns an unexpected response."""


class CredentialBrokerClient:
    """Wraps the credential-broker-service REST API for LLM API key storage."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def store_api_key(self, api_key: str, label: str, user_id: str) -> str:
        """POST the key to credential-broker; return the UUID cred_id string.

        The plaintext api_key is never logged here or anywhere downstream.
        """
        payload = {
            "tool_id": str(_LLM_CONFIG_TOOL_ID),
            "user_id": user_id,
            "name": label,
            "provider": "llm-config",
            "cred_type": "api_key",
            "credential": {"value": api_key},
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{self._base_url}/credentials/", json=payload)
        if resp.status_code not in (200, 201):
            raise CredentialBrokerError(
                f"credential-broker store failed: {resp.status_code}"
            )
        return str(resp.json()["id"])

    async def retrieve_api_key(self, cred_id: str) -> str:
        """GET the credential by ID and return the plaintext key.

        Raises CredentialBrokerError if not found or broker unavailable.
        Never logs the returned key value.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self._base_url}/credentials/{cred_id}")
        if resp.status_code == 404:
            raise CredentialBrokerError(f"Credential {cred_id!r} not found in broker")
        if resp.status_code != 200:
            raise CredentialBrokerError(
                f"credential-broker retrieve failed: {resp.status_code}"
            )
        data = resp.json()
        key = data.get("credential", {}).get("value")
        if not key:
            raise CredentialBrokerError(
                f"Credential {cred_id!r} has no 'value' field in credential dict"
            )
        return key

    async def delete_credential(self, cred_id: str) -> None:
        """DELETE the credential from the broker. No-op if already absent."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(f"{self._base_url}/credentials/{cred_id}")
        if resp.status_code not in (200, 204, 404):
            raise CredentialBrokerError(
                f"credential-broker delete failed: {resp.status_code}"
            )
