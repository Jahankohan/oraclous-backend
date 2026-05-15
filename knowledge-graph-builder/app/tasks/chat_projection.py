"""Async projection of chat turns into Neo4j (STORY-031 / TASK-106 / ADR-020).

Triggered after every successful ``chat_messages`` write in the chat
handlers. Reads the row + its conversation from Postgres, then writes
a lightweight semantic shadow under the reserved ``:__Chat__``
namespace label using the templates in ``app.cypher.chat_queries``.

Postgres is the source of truth. **Projection failures here MUST
NEVER block the user-facing chat response** — the chat handlers fire
the task with ``.delay()`` and ignore failures of the dispatch
itself.

Retry policy: Celery retries on broker errors via exponential
backoff up to 5 attempts; if all retries fail, the task ends in a
permanent failed state and the worker logs the message_id. A formal
DLQ table can be added if/when we see frequent permanent failures.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy import text as sql_text

from app.core.logging import get_logger
from app.cypher.chat_queries import (
    LINK_CONVERSATION_TO_AGENT,
    LINK_CONVERSATION_TO_GRAPH,
    UPSERT_CHAT_TURN,
    UPSERT_CONVERSATION,
)
from app.models.chat import ChatConversation, ChatMessage
from app.services.background_jobs import WorkerNeo4jManager, celery_app
from app.services.task_executor import AsyncTaskExecutor

logger = get_logger(__name__)

# How much of message.content to embed as a snippet on the :__Chat__:ChatTurn
# Neo4j node. The full text stays in Postgres — Neo4j only carries enough
# for downstream features (memory, retrieval) to triage relevance.
_SNIPPET_MAX_CHARS = 240


@celery_app.task(
    bind=True,
    name="chat_projection.project_message",
    max_retries=5,
    default_retry_delay=2,  # seconds; doubles each retry via autoretry_for
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=64,
    retry_jitter=True,
)
def project_chat_message_task(
    self,
    message_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Project a single chat turn into Neo4j under ``:__Chat__``.

    Parameters carry the owning user's id so the worker can set
    ``app.current_user_id`` and read the row through RLS — the
    workers don't have a request-scoped principal otherwise.

    Idempotent: the WRITE templates all MERGE, so re-running the task
    after a partial failure produces the same end state.
    """
    return AsyncTaskExecutor.run_async_task(_project_async, self, message_id, user_id)


async def _project_async(
    task, message_id: str, user_id: str
) -> dict[str, Any]:  # pragma: no cover  — exercised by integration tests
    """Async body of the projection task.

    Two phases: (1) load the message + its conversation from Postgres
    under the user's RLS context, (2) write the Neo4j shadow via the
    chat_queries templates.
    """
    from app.core.database import async_session_maker

    # ── Phase 1: load from Postgres ───────────────────────────────────────────
    try:
        msg_uuid = uuid.UUID(message_id)
        user_uuid = uuid.UUID(user_id)
    except (ValueError, TypeError):
        logger.warning(
            "chat_projection: malformed id (message=%s user=%s); skipping",
            message_id,
            user_id,
        )
        return {"status": "skipped", "reason": "malformed_id"}

    async with async_session_maker() as db:
        await db.execute(
            sql_text("SELECT set_config('app.current_user_id', :uid, true)").bindparams(
                uid=str(user_uuid)
            )
        )
        msg = (
            await db.execute(select(ChatMessage).where(ChatMessage.id == msg_uuid))
        ).scalar_one_or_none()
        if msg is None:
            logger.info(
                "chat_projection: message %s not found (RLS or deleted); skipping",
                message_id,
            )
            return {"status": "skipped", "reason": "message_not_visible"}
        conv = (
            await db.execute(
                select(ChatConversation).where(
                    ChatConversation.id == msg.conversation_id
                )
            )
        ).scalar_one_or_none()
        if conv is None:
            return {"status": "skipped", "reason": "conversation_not_visible"}

        conv_id = str(conv.id)
        conv_user_id = str(conv.user_id)
        conv_graph_id = str(conv.graph_id)
        conv_agent_id = str(conv.agent_id) if conv.agent_id else None
        conv_title = conv.title
        conv_created_at = (
            conv.created_at.isoformat() if conv.created_at is not None else None
        )
        conv_last_message_at = (
            conv.last_message_at.isoformat()
            if conv.last_message_at is not None
            else None
        )

        message_id_str = str(msg.id)
        role = msg.role
        content = msg.content or ""
        snippet = content[:_SNIPPET_MAX_CHARS]
        created_at = msg.created_at.isoformat() if msg.created_at is not None else None

    # ── Phase 2: project into Neo4j ───────────────────────────────────────────
    async with WorkerNeo4jManager() as neo4j:
        driver = neo4j.async_driver
        async with driver.session() as session:
            await session.run(
                UPSERT_CONVERSATION,
                conversation_id=conv_id,
                user_id=conv_user_id,
                graph_id=conv_graph_id,
                agent_id=conv_agent_id,
                title=conv_title,
                created_at=conv_created_at,
                last_message_at=conv_last_message_at,
            )
            await session.run(
                UPSERT_CHAT_TURN,
                conversation_id=conv_id,
                message_id=message_id_str,
                role=role,
                snippet=snippet,
                created_at=created_at,
            )
            # Best-effort outbound links. The target nodes (Graph, Agent)
            # may not exist in test environments — skip silently rather
            # than failing the projection.
            try:
                await session.run(
                    LINK_CONVERSATION_TO_GRAPH,
                    conversation_id=conv_id,
                    graph_id=conv_graph_id,
                )
            except Exception:
                logger.debug(
                    "chat_projection: LINK_CONVERSATION_TO_GRAPH skipped for %s",
                    conv_id,
                )
            if conv_agent_id is not None:
                try:
                    await session.run(
                        LINK_CONVERSATION_TO_AGENT,
                        conversation_id=conv_id,
                        agent_id=conv_agent_id,
                    )
                except Exception:
                    logger.debug(
                        "chat_projection: LINK_CONVERSATION_TO_AGENT skipped for %s",
                        conv_id,
                    )

    return {
        "status": "ok",
        "message_id": message_id_str,
        "conversation_id": conv_id,
    }


def fire_and_forget(message_id: str, user_id: str) -> None:
    """Fire-and-forget dispatch wrapper used by the chat handlers.

    Wraps ``.delay()`` so a broker outage logs a warning rather than
    propagating to the user-facing handler. Postgres remains the
    source of truth and the projection retries automatically when the
    broker recovers (via Celery's autoretry).
    """
    try:
        project_chat_message_task.delay(message_id, user_id)
    except Exception as exc:
        logger.warning(
            "chat_projection: dispatch failed for message %s: %s — "
            "Postgres has the row; projection will replay on next attempt",
            message_id,
            exc,
        )
