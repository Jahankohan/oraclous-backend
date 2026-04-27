"""Connector CRUD and sync service (ORA-78)."""

from __future__ import annotations

import hmac as hmac_lib
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.graph import Connector, ConnectorSyncLog, WebhookEvent
from app.schemas.connector_schemas import (
    ConnectorResponse,
    RegisterConnectorRequest,
    SyncLogResponse,
    UpdateConnectorRequest,
)

logger = get_logger(__name__)


class ConnectorService:
    """Manages connector lifecycle: register, list, update, delete, trigger sync."""

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def register_connector(
        self,
        db: AsyncSession,
        graph_id: str,
        user_id: str,
        request: RegisterConnectorRequest,
    ) -> ConnectorResponse:
        connector = Connector(
            graph_id=graph_id,
            user_id=user_id,
            name=request.name,
            connector_type=request.connector_type,
            status="active",
            config=request.config.model_dump(),
            schedule=request.schedule,
        )
        db.add(connector)
        await db.commit()
        await db.refresh(connector)
        logger.info(
            f"Registered connector {connector.id} ({request.connector_type}) for graph {graph_id}"
        )
        return self._to_response(connector)

    async def list_connectors(
        self,
        db: AsyncSession,
        graph_id: str,
        user_id: str,
    ) -> list[ConnectorResponse]:
        result = await db.execute(
            select(Connector).where(
                Connector.graph_id == graph_id,
                Connector.user_id == user_id,
            )
        )
        connectors = result.scalars().all()
        return [self._to_response(c) for c in connectors]

    async def get_connector(
        self,
        db: AsyncSession,
        graph_id: str,
        user_id: str,
        connector_id: str,
    ) -> ConnectorResponse:
        connector = await self._get_or_404(db, graph_id, user_id, connector_id)
        return self._to_response(connector)

    async def update_connector(
        self,
        db: AsyncSession,
        graph_id: str,
        user_id: str,
        connector_id: str,
        request: UpdateConnectorRequest,
    ) -> ConnectorResponse:
        connector = await self._get_or_404(db, graph_id, user_id, connector_id)
        if request.name is not None:
            connector.name = request.name
        if request.config is not None:
            connector.config = request.config.model_dump()
        if request.schedule is not None:
            connector.schedule = request.schedule
        if request.status is not None:
            connector.status = request.status
        connector.updated_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(connector)
        return self._to_response(connector)

    async def delete_connector(
        self,
        db: AsyncSession,
        graph_id: str,
        user_id: str,
        connector_id: str,
    ) -> None:
        connector = await self._get_or_404(db, graph_id, user_id, connector_id)
        await db.delete(connector)
        await db.commit()
        logger.info(f"Deleted connector {connector_id} for graph {graph_id}")

    # ------------------------------------------------------------------
    # Sync trigger
    # ------------------------------------------------------------------

    async def trigger_sync(
        self,
        db: AsyncSession,
        graph_id: str,
        user_id: str,
        connector_id: str,
    ) -> dict[str, Any]:
        """Enqueue a Celery sync task for the connector."""
        connector = await self._get_or_404(db, graph_id, user_id, connector_id)
        if connector.connector_type == "webhook_receiver":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="webhook_receiver connectors are push-only — use the webhook URL instead",
            )
        from app.services.connector_jobs import sync_connector

        task = sync_connector.delay(str(connector.id))
        logger.info(f"Queued sync for connector {connector_id}, task_id={task.id}")
        return {"connector_id": connector_id, "task_id": task.id, "status": "queued"}

    # ------------------------------------------------------------------
    # Sync logs
    # ------------------------------------------------------------------

    async def list_sync_logs(
        self,
        db: AsyncSession,
        graph_id: str,
        user_id: str,
        connector_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        # Verify ownership
        await self._get_or_404(db, graph_id, user_id, connector_id)

        result = await db.execute(
            select(ConnectorSyncLog)
            .where(ConnectorSyncLog.connector_id == UUID(connector_id))
            .order_by(ConnectorSyncLog.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        logs = result.scalars().all()
        count_result = await db.execute(
            select(func.count()).where(
                ConnectorSyncLog.connector_id == UUID(connector_id)
            )
        )
        total = count_result.scalar() or 0
        return {
            "logs": [SyncLogResponse.model_validate(log) for log in logs],
            "total": total,
        }

    # ------------------------------------------------------------------
    # Webhook event ingestion
    # ------------------------------------------------------------------

    async def store_webhook_event(
        self,
        db: AsyncSession,
        connector_id: str,
        payload: dict[str, Any],
        payload_hash: str,
        event_type: str | None = None,
    ) -> str:
        """Persist a webhook event and return its UUID."""
        event = WebhookEvent(
            connector_id=UUID(connector_id),
            event_type=event_type,
            payload_hash=payload_hash,
            payload=payload,
            status="pending",
        )
        db.add(event)
        await db.commit()
        await db.refresh(event)
        return str(event.id)

    async def is_duplicate_event(
        self,
        db: AsyncSession,
        connector_id: str,
        payload_hash: str,
    ) -> bool:
        result = await db.execute(
            select(WebhookEvent).where(
                WebhookEvent.connector_id == UUID(connector_id),
                WebhookEvent.payload_hash == payload_hash,
            )
        )
        return result.scalars().first() is not None

    async def load_connector_by_ids(
        self,
        db: AsyncSession,
        graph_id: str,
        connector_id: str,
    ) -> Connector:
        """Load connector by graph_id + connector_id (no user_id — for webhook path)."""
        result = await db.execute(
            select(Connector).where(
                Connector.id == UUID(connector_id),
                Connector.graph_id == graph_id,
            )
        )
        connector = result.scalars().first()
        if not connector:
            # Return 404 — never 403 — to prevent graph/connector enumeration (spec §9)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found"
            )
        return connector

    # ------------------------------------------------------------------
    # HMAC verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify_hmac(
        raw_body: bytes,
        auth_config: dict[str, Any],
        headers: dict[str, str],
        credential_value: str | None = None,
    ) -> bool:
        """
        Verify HMAC-SHA256 webhook signature.

        Supports GitHub (X-Hub-Signature-256), Stripe (Stripe-Signature),
        and generic HMAC patterns.
        """
        if not credential_value:
            logger.warning("HMAC verification skipped: no credential resolved")
            return False

        algorithm = auth_config.get("hmac_algorithm", "sha256")
        header_name = auth_config.get("hmac_header", "X-Hub-Signature-256")
        import hashlib as _hashlib

        digest = hmac_lib.new(
            credential_value.encode(),
            raw_body,
            _hashlib.sha256,
        ).hexdigest()
        expected_sig = f"{algorithm}={digest}"

        # Case-insensitive header lookup
        received_sig = ""
        for key, val in headers.items():
            if key.lower() == header_name.lower():
                received_sig = val
                break

        return hmac_lib.compare_digest(expected_sig, received_sig)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_or_404(
        self,
        db: AsyncSession,
        graph_id: str,
        user_id: str,
        connector_id: str,
    ) -> Connector:
        try:
            uid = UUID(connector_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found"
            ) from None

        result = await db.execute(
            select(Connector).where(
                Connector.id == uid,
                Connector.graph_id == graph_id,
                Connector.user_id == user_id,
            )
        )
        connector = result.scalars().first()
        if not connector:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found"
            )
        return connector

    def _to_response(self, connector: Connector) -> ConnectorResponse:
        webhook_url: str | None = None
        if connector.connector_type == "webhook_receiver":
            webhook_url = f"{settings.SERVICE_URL}/api/v1/webhooks/{connector.graph_id}/{connector.id}"
        return ConnectorResponse(
            id=str(connector.id),
            graph_id=connector.graph_id,
            name=connector.name,
            connector_type=connector.connector_type,
            status=connector.status,
            schedule=connector.schedule,
            last_synced_at=connector.last_synced_at,
            webhook_url=webhook_url,
            created_at=connector.created_at,
            updated_at=connector.updated_at,
        )


connector_service = ConnectorService()
