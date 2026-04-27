"""Webhook receiver endpoint (ORA-78).

POST /api/v1/webhooks/{graph_id}/{connector_id}

- No JWT auth — caller authenticates via HMAC signature
- Validates HMAC before any processing
- Deduplicates by payload hash
- Dispatches to Celery for async KG ingestion
"""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_database
from app.core.logging import get_logger
from app.core.rate_limiter import limiter
from app.services.connector_service import connector_service

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/webhooks/{graph_id}/{connector_id}",
    summary="Receive an inbound webhook event",
    response_model=dict[str, Any],
    responses={429: {"description": "Rate limit exceeded"}},
)
@limiter.limit("100/minute")
async def receive_webhook(
    graph_id: str,
    connector_id: str,
    request: Request,
    db: AsyncSession = Depends(get_database),
) -> dict[str, Any]:
    """
    Inbound webhook receiver for push-based connectors.

    Processing pipeline (spec §6.2):
    1. Load connector — verify graph_id + connector_id match
    2. Read raw body BEFORE any parsing (HMAC requires raw bytes)
    3. HMAC-SHA256 verification
    4. Deduplication check
    5. Store event (status=pending)
    6. Dispatch to Celery for async KG ingestion
    """
    # 1. Load connector — 404 on mismatch (never 403; prevents enumeration)
    connector = await connector_service.load_connector_by_ids(
        db, graph_id, connector_id
    )
    if connector.connector_type != "webhook_receiver":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found"
        )

    # 2. Read raw body
    raw_body = await request.body()

    # 3. HMAC verification
    config = connector.config or {}
    auth_config = config.get("auth", {})
    if auth_config.get("auth_type") == "hmac_secret":
        hmac_cred_id = auth_config.get("hmac_secret_credential_id")
        resolved_secret = await _resolve_credential(hmac_cred_id)
        headers_dict = dict(request.headers)
        if not connector_service.verify_hmac(
            raw_body, auth_config, headers_dict, resolved_secret
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature",
            )

    # 4. Parse payload + dedup
    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        payload = {"_raw": raw_body.decode(errors="replace")}

    payload_hash = hashlib.sha256(raw_body).hexdigest()

    if await connector_service.is_duplicate_event(db, connector_id, payload_hash):
        return {"status": "duplicate", "accepted": False}

    # 5. Store event (status=pending)
    event_type = request.headers.get("X-GitHub-Event") or request.headers.get(
        "X-Event-Type"
    )
    event_id = await connector_service.store_webhook_event(
        db, connector_id, payload, payload_hash, event_type=event_type
    )

    # 6. Dispatch to Celery
    try:
        from app.services.connector_jobs import process_webhook_event

        process_webhook_event.delay(event_id, graph_id, connector_id)
    except Exception as e:
        logger.error(f"Failed to dispatch webhook event {event_id} to Celery: {e}")
        # Event is stored; retry worker will pick it up

    logger.info(
        f"Accepted webhook event {event_id} for connector {connector_id} graph {graph_id}"
    )
    return {"status": "accepted", "event_id": event_id}


async def _resolve_credential(credential_id: str | None) -> str | None:
    """Resolve a credential from the credential broker service."""
    if not credential_id:
        return None
    try:
        from app.services.credential_service import credential_service

        creds = await credential_service.get_user_credentials(
            user_id="system", provider=credential_id
        )
        return (creds or {}).get("access_token") or (creds or {}).get("secret")
    except Exception as e:
        logger.error(f"Failed to resolve credential {credential_id}: {e}")
        return None
