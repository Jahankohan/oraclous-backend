"""Celery tasks for connector sync and webhook processing (ORA-78).

Dual-driver rule:
- These are Celery worker tasks → use task-scoped sync Neo4j driver (NullPool)
- Never import the FastAPI async_driver here

Beat schedule entries are added to background_jobs.py.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import get_logger
from app.models.graph import Connector, ConnectorSyncLog, WebhookEvent
from app.services.background_jobs import celery_app

logger = get_logger(__name__)

# Task-scoped async engine for Celery workers (NullPool — no connection reuse between tasks)
_worker_engine = create_async_engine(
    settings.POSTGRES_URL,
    poolclass=NullPool,
    echo=False,
    future=True,
)
_worker_session_maker = async_sessionmaker(
    bind=_worker_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ---------------------------------------------------------------------------
# Webhook event → KG mappers
# ---------------------------------------------------------------------------


class PassthroughMapper:
    """Default mapper: JSON-dump the payload and pass through pipeline."""

    def to_text(self, payload: dict[str, Any], context_hint: str = "") -> str:
        import json

        prefix = f"Context: {context_hint}\n\n" if context_hint else ""
        return prefix + json.dumps(payload, default=str, ensure_ascii=False)


class GithubEventMapper:
    """Maps GitHub webhook push/issue/PR events to text summaries."""

    def to_text(self, payload: dict[str, Any], context_hint: str = "") -> str:
        event_type = payload.get("action") or payload.get("event_type", "")
        repo = (payload.get("repository") or {}).get("full_name", "unknown")

        if "issue" in payload:
            issue = payload["issue"]
            return (
                f"GitHub issue event in {repo}: {event_type}\n"
                f"Issue #{issue.get('number')}: {issue.get('title')}\n"
                f"Author: {(issue.get('user') or {}).get('login', 'unknown')}\n"
                f"Body: {(issue.get('body') or '')[:400]}"
            )
        if "pull_request" in payload:
            pr = payload["pull_request"]
            return (
                f"GitHub pull request event in {repo}: {event_type}\n"
                f"PR #{pr.get('number')}: {pr.get('title')}\n"
                f"Author: {(pr.get('user') or {}).get('login', 'unknown')}\n"
                f"Body: {(pr.get('body') or '')[:400]}"
            )
        if "commits" in payload:
            commits = payload["commits"][:5]
            commit_lines = "\n".join(
                f"- {c.get('id', '')[:8]}: {c.get('message', '')[:100]}"
                for c in commits
            )
            return f"GitHub push to {repo}:\n{commit_lines}"

        import json

        return f"GitHub event ({event_type}) in {repo}:\n{json.dumps(payload, default=str)[:500]}"


EVENT_MAPPERS = {
    "github": GithubEventMapper,
    "webhook_receiver": PassthroughMapper,
}


# ---------------------------------------------------------------------------
# Helper: run async code in Celery sync context
# ---------------------------------------------------------------------------


def _run(coro):
    """Execute a coroutine in a new event loop (Celery workers are sync)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Task: poll_due_connectors — master scheduler (every 60s via beat)
# ---------------------------------------------------------------------------


@celery_app.task(name="connectors.poll_due_connectors")
def poll_due_connectors() -> dict[str, Any]:
    """
    Master scheduler: find all active, scheduled connectors whose next sync is due,
    and dispatch individual sync tasks.

    Runs every 60 seconds via Celery Beat.
    """
    return _run(_poll_due_connectors_async())


async def _poll_due_connectors_async() -> dict[str, Any]:
    dispatched: list[str] = []

    async with _worker_session_maker() as db:
        # Find connectors that are active, have a schedule, and are due for sync.
        # "Due" = last_synced_at IS NULL (never synced) OR last_synced_at + interval <= now
        # We use a simple heuristic: any active scheduled connector not synced in the last
        # hour is considered due. The proper approach would parse the cron expression
        # (not implemented here to avoid a heavy dependency; operators can use manual sync).
        result = await db.execute(
            select(Connector).where(
                Connector.status == "active",
                Connector.schedule.isnot(None),
                Connector.connector_type != "webhook_receiver",
            )
        )
        connectors = result.scalars().all()

        now = datetime.now(UTC)
        for connector in connectors:
            if _is_sync_due(connector, now):
                sync_connector.delay(str(connector.id))
                dispatched.append(str(connector.id))

    logger.info(f"poll_due_connectors: dispatched {len(dispatched)} sync tasks")
    return {"dispatched": len(dispatched), "connector_ids": dispatched}


def _is_sync_due(connector: Connector, now: datetime) -> bool:
    """Simple due-check: sync if never synced or last sync was > 1h ago."""
    if connector.last_synced_at is None:
        return True
    last = connector.last_synced_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    return (now - last).total_seconds() >= 3600


# ---------------------------------------------------------------------------
# Task: sync_connector — execute one full sync cycle
# ---------------------------------------------------------------------------


@celery_app.task(name="connectors.sync_connector", bind=True, max_retries=3)
def sync_connector(self, connector_id: str) -> dict[str, Any]:
    """
    Execute one full sync cycle for a connector.

    - Loads connector config
    - Fetches items via appropriate ConnectorFetcher
    - Ingests batches through pipeline_service.process_documents()
    - Updates last_synced_at and cursor
    - Logs the sync result in connector_sync_logs
    """
    try:
        return _run(_sync_connector_async(connector_id))
    except Exception as exc:
        logger.error(f"sync_connector {connector_id} failed: {exc}")
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1)) from None


async def _sync_connector_async(connector_id: str) -> dict[str, Any]:
    from app.schemas.graph_schemas import IngestMode
    from app.services.connectors.base import ConnectorFetcher
    from app.services.pipeline_service import pipeline_service

    log_id: str | None = None
    items_processed = 0
    entities_extracted = 0
    sync_status = "error"
    error_message: str | None = None

    async with _worker_session_maker() as db:
        # Load connector
        result = await db.execute(
            select(Connector).where(Connector.id == UUID(connector_id))
        )
        connector = result.scalars().first()
        if not connector:
            logger.warning(f"sync_connector: connector {connector_id} not found")
            return {"status": "not_found"}

        if connector.status != "active":
            logger.info(
                f"sync_connector: connector {connector_id} is {connector.status}, skipping"
            )
            return {"status": "skipped", "reason": connector.status}

        # Create sync log entry
        log = ConnectorSyncLog(connector_id=UUID(connector_id), status="running")
        db.add(log)
        await db.commit()
        await db.refresh(log)
        log_id = str(log.id)

        try:
            # Fetch items
            fetcher = ConnectorFetcher.for_type(
                connector.connector_type, connector.config or {}
            )
            items = fetcher.fetch_since(connector.last_sync_cursor)
            items_processed = len(items)

            # Ingest in batches of 20 items each
            batch_size = 20
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                text = fetcher.to_text(batch)
                if not text.strip():
                    continue

                context_hint = (connector.config.get("entity_mapping") or {}).get(
                    "context_hint", ""
                )
                source_label = f"connector:{connector.connector_type}:{connector.name}"

                result_docs = await pipeline_service.process_documents(
                    documents=[
                        {"text": text, "source": source_label, "context": context_hint}
                    ],
                    graph_id=UUID(connector.graph_id),
                    user_id=connector.user_id,
                    mode=IngestMode.INCREMENTAL,
                )
                entities_extracted += result_docs.get("extracted_entities", 0)

            # Update connector cursor and last_synced_at
            connector.last_sync_cursor = fetcher.last_cursor
            connector.last_synced_at = datetime.now(UTC)
            sync_status = "success"

        except Exception as exc:
            error_message = str(exc)
            logger.error(f"sync_connector {connector_id} ingestion error: {exc}")
            sync_status = "error"

        # Finalize sync log
        await db.execute(
            update(ConnectorSyncLog)
            .where(ConnectorSyncLog.id == log.id)
            .values(
                finished_at=datetime.now(UTC),
                status=sync_status,
                items_processed=items_processed,
                entities_extracted=entities_extracted,
                error_message=error_message,
            )
        )
        await db.commit()

    return {
        "connector_id": connector_id,
        "log_id": log_id,
        "status": sync_status,
        "items_processed": items_processed,
        "entities_extracted": entities_extracted,
    }


# ---------------------------------------------------------------------------
# Task: process_webhook_event — ingest a single webhook event
# ---------------------------------------------------------------------------


@celery_app.task(name="connectors.process_webhook_event", bind=True, max_retries=5)
def process_webhook_event(
    self, event_id: str, graph_id: str, connector_id: str
) -> dict[str, Any]:
    """
    Process a single webhook event through the KG ingestion pipeline.

    Retry schedule (exponential backoff): 1m, 5m, 30m, 2h, 24h
    """
    backoff_seconds = [60, 300, 1800, 7200, 86400]
    try:
        return _run(_process_webhook_event_async(event_id, graph_id, connector_id))
    except Exception as exc:
        retry_index = self.request.retries
        countdown = backoff_seconds[min(retry_index, len(backoff_seconds) - 1)]
        logger.error(
            f"process_webhook_event {event_id} failed (attempt {retry_index + 1}): {exc}"
        )
        raise self.retry(exc=exc, countdown=countdown) from None


async def _process_webhook_event_async(
    event_id: str,
    graph_id: str,
    connector_id: str,
) -> dict[str, Any]:
    from app.schemas.graph_schemas import IngestMode
    from app.services.pipeline_service import pipeline_service

    async with _worker_session_maker() as db:
        # Load event
        result = await db.execute(
            select(WebhookEvent).where(WebhookEvent.id == UUID(event_id))
        )
        event = result.scalars().first()
        if not event:
            logger.warning(f"process_webhook_event: event {event_id} not found")
            return {"status": "not_found"}

        # Load connector
        conn_result = await db.execute(
            select(Connector).where(Connector.id == UUID(connector_id))
        )
        connector = conn_result.scalars().first()
        if not connector:
            await db.execute(
                update(WebhookEvent)
                .where(WebhookEvent.id == UUID(event_id))
                .values(status="error", error_message="connector not found")
            )
            await db.commit()
            return {"status": "error", "reason": "connector not found"}

        try:
            mapper_class = EVENT_MAPPERS.get(
                connector.connector_type, PassthroughMapper
            )
            mapper = mapper_class()
            context_hint = (connector.config.get("entity_mapping") or {}).get(
                "context_hint", ""
            )
            text = mapper.to_text(event.payload, context_hint)

            if text.strip():
                source_label = f"connector:{connector.connector_type}:{connector.name}"
                await pipeline_service.process_documents(
                    documents=[
                        {"text": text, "source": source_label, "context": context_hint}
                    ],
                    graph_id=UUID(graph_id),
                    user_id=connector.user_id,
                    mode=IngestMode.INCREMENTAL,
                )

            await db.execute(
                update(WebhookEvent)
                .where(WebhookEvent.id == UUID(event_id))
                .values(status="processed", processed_at=datetime.now(UTC))
            )
            await db.commit()

            logger.info(
                f"Processed webhook event {event_id} for connector {connector_id}"
            )
            return {"event_id": event_id, "status": "processed"}

        except Exception as exc:
            await db.execute(
                update(WebhookEvent)
                .where(WebhookEvent.id == UUID(event_id))
                .values(status="error", error_message=str(exc))
            )
            await db.commit()
            raise


# ---------------------------------------------------------------------------
# Task: retry_failed_events — exponential backoff retry for errored webhooks
# ---------------------------------------------------------------------------


@celery_app.task(name="connectors.retry_failed_events")
def retry_failed_events() -> dict[str, Any]:
    """
    Re-dispatch webhook events stuck in 'error' status.
    Runs every 5 minutes via Celery Beat.
    """
    return _run(_retry_failed_events_async())


async def _retry_failed_events_async() -> dict[str, Any]:
    dispatched = 0
    async with _worker_session_maker() as db:
        result = await db.execute(
            select(WebhookEvent).where(WebhookEvent.status == "error").limit(50)
        )
        events = result.scalars().all()
        for event in events:
            conn_result = await db.execute(
                select(Connector).where(Connector.id == event.connector_id)
            )
            connector = conn_result.scalars().first()
            if connector:
                process_webhook_event.delay(
                    str(event.id), connector.graph_id, str(connector.id)
                )
                dispatched += 1

    return {"dispatched": dispatched}
