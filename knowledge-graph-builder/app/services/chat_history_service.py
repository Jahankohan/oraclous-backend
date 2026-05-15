"""Chat history persistence service (STORY-031 / TASK-103).

Writes user + assistant turns and per-tool audit rows into the
Postgres tables owned by ADR-020. Callers are the three chat
endpoints in ``app/api/v1/endpoints/{chat,agents}.py``.

Caller contract:

* Pass an ``AsyncSession`` obtained from ``get_chat_db`` so the RLS
  GUC ``app.current_user_id`` is set on the connection. Without it,
  RLS policies will reject every write.
* The session is the source-of-truth transaction boundary. The
  service flushes but does **not** commit — the caller commits once
  the HTTP response is ready, so a failure mid-response rolls back
  the entire turn (user + assistant + tool calls) atomically.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import zstandard as zstd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import (
    ChatConversation,
    ChatMessage,
    ChatMessageToolCall,
)
from app.services.llm_pricing import estimate_cost_usd

logger = logging.getLogger(__name__)

# Caps for tool result persistence (ADR-020 / STORY-031 decisions table).
MAX_RAW_BYTES = 50 * 1024 * 1024  # 50 MB pre-compression
MAX_COMPRESSED_BYTES = 5 * 1024 * 1024  # 5 MB compressed cap

_ZSTD_LEVEL = 3
_compressor = zstd.ZstdCompressor(level=_ZSTD_LEVEL)


class ChatHistoryService:
    """Stateless write-side persistence for chat turns.

    Methods take an explicit ``db`` session because RLS depends on the
    ``app.current_user_id`` GUC being set on that exact connection
    (see ``get_chat_db`` in ``app/api/dependencies.py``).
    """

    async def get_or_create_conversation(
        self,
        db: AsyncSession,
        *,
        user_id: str,
        graph_id: str,
        agent_id: str | None,
        conversation_id: str | None,
        first_message: str | None,
    ) -> ChatConversation:
        """Resolve the conversation for this turn.

        Three cases:

        1. ``conversation_id`` is provided AND belongs to ``user_id`` →
           return the row. RLS prevents cross-user reads here.
        2. ``conversation_id`` is provided but the row is missing
           (deleted or never existed for this user) → create a new
           conversation. We do not 404 on the write path because doing
           so would lose the user's message; clients can paginate to
           discover the new id from the response.
        3. ``conversation_id`` is None → always create a new
           conversation, titled from ``first_message`` (truncated to 80
           chars; falls back to "New chat" if no first message).
        """
        if conversation_id is not None:
            try:
                cid = uuid.UUID(conversation_id)
            except (ValueError, TypeError):
                # Malformed id: treat as new conversation.
                conversation_id = None
            else:
                stmt = select(ChatConversation).where(
                    ChatConversation.id == cid,
                    ChatConversation.user_id == uuid.UUID(user_id),
                    ChatConversation.deleted_at.is_(None),
                )
                result = await db.execute(stmt)
                existing = result.scalar_one_or_none()
                if existing is not None:
                    return existing
                conversation_id = None  # fall through to creation

        # Build a friendly title from the user's first message.
        if first_message:
            title = first_message.strip()[:80] or "New chat"
        else:
            title = "New chat"

        conv = ChatConversation(
            user_id=uuid.UUID(user_id),
            graph_id=uuid.UUID(graph_id),
            agent_id=uuid.UUID(agent_id) if agent_id else None,
            title=title,
        )
        db.add(conv)
        await db.flush()
        return conv

    async def write_user_message(
        self,
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        content: str,
    ) -> ChatMessage:
        """Persist a user turn. Returns the freshly inserted row."""
        msg = ChatMessage(
            conversation_id=conversation_id,
            role="user",
            content=content,
        )
        db.add(msg)
        await db.flush()
        return msg

    async def write_assistant_message(
        self,
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        content: str,
        model: str | None = None,
        provider: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int | None = None,
        reasoning_mode: str | None = None,
        retriever_used: str | None = None,
        error: str | None = None,
        cancelled: bool = False,
        sources: list[dict] | None = None,
    ) -> ChatMessage:
        """Persist an assistant turn with audit metadata.

        ``cost_usd`` is computed from ``model`` + token counts via the
        per-model price table. Unknown models leave it ``NULL``.
        """
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        msg = ChatMessage(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            reasoning_mode=reasoning_mode,
            retriever_used=retriever_used,
            error=error,
            cancelled=cancelled,
            sources=sources,
        )
        db.add(msg)
        await db.flush()
        # Bump conversation last_message_at via touch.
        stmt = select(ChatConversation).where(ChatConversation.id == conversation_id)
        conv = (await db.execute(stmt)).scalar_one_or_none()
        if conv is not None:
            # SQLAlchemy onupdate fires on actual UPDATE — we trigger one
            # by re-assigning the title to itself. Simpler than a raw
            # UPDATE statement and keeps everything in the same session.
            conv.title = conv.title
            await db.flush()
        return msg

    async def write_tool_call(
        self,
        db: AsyncSession,
        *,
        message_id: uuid.UUID,
        sequence_index: int,
        tool_name: str,
        args: dict[str, Any] | None,
        result: Any,
        latency_ms: int | None = None,
        error: str | None = None,
    ) -> ChatMessageToolCall:
        """Persist a tool-invocation row.

        ``result`` is serialized to JSON then zstd-compressed. If the
        raw JSON exceeds ``MAX_RAW_BYTES`` (50 MB) it is replaced with
        a placeholder before compression. If the compressed payload
        still exceeds ``MAX_COMPRESSED_BYTES`` (5 MB), the row is
        marked truncated and the compressed blob is discarded.
        """
        compressed, compression, uncompressed_size, truncated, summary = (
            _compress_result(result)
        )
        row = ChatMessageToolCall(
            message_id=message_id,
            sequence_index=sequence_index,
            tool_name=tool_name,
            args_json=args or {},
            result_summary=summary,
            result_compressed=compressed,
            result_compression=compression,
            result_uncompressed_size_bytes=uncompressed_size,
            result_truncated=truncated,
            latency_ms=latency_ms,
            error=error,
        )
        db.add(row)
        await db.flush()
        return row

    async def update_message_for_cancel(
        self,
        db: AsyncSession,
        *,
        message_id: uuid.UUID,
        partial_content: str,
    ) -> None:
        """Mark an assistant turn cancelled and persist whatever streamed.

        Idempotent — re-calling with the same id is a no-op aside from
        overwriting partial_content with the latest version.
        """
        stmt = select(ChatMessage).where(ChatMessage.id == message_id)
        msg = (await db.execute(stmt)).scalar_one_or_none()
        if msg is None:
            logger.warning(
                "cancel-update: message %s not found; skipping",
                message_id,
            )
            return
        msg.content = partial_content
        msg.cancelled = True
        await db.flush()


def _compress_result(
    result: Any,
) -> tuple[bytes | None, str | None, int | None, bool, str | None]:
    """Serialize + compress a tool result. Returns the columns the row needs.

    Returns ``(compressed_bytes, compression_label, uncompressed_size,
    truncated, summary)``. ``summary`` is a short human-readable note
    for the row's ``result_summary`` column.
    """
    if result is None:
        return None, None, None, False, None

    # ``default=str`` makes datetime/UUID/Decimal/Path round-trippable.
    # For exotic objects the json encoder still falls back to their str()
    # representation rather than raising. The result is best-effort
    # audit — for tool results we own, this is good enough.
    raw = json.dumps(result, default=str).encode("utf-8")
    uncompressed_size = len(raw)
    truncated = False

    if uncompressed_size > MAX_RAW_BYTES:
        placeholder = json.dumps(
            {"_truncated": True, "_orig_size": uncompressed_size}
        ).encode("utf-8")
        raw = placeholder
        truncated = True

    compressed = _compressor.compress(raw)
    if len(compressed) > MAX_COMPRESSED_BYTES:
        # Cannot store; drop the blob, keep the size metadata.
        return (
            None,
            None,
            uncompressed_size,
            True,
            _summarize_result(result, uncompressed_size, truncated=True),
        )

    return (
        compressed,
        "zstd",
        uncompressed_size,
        truncated,
        _summarize_result(result, uncompressed_size, truncated),
    )


def _summarize_result(
    result: Any, uncompressed_size: int, truncated: bool
) -> str | None:
    """Build a short human-readable summary for ``result_summary``."""
    if truncated:
        return f"truncated · {uncompressed_size} bytes raw"
    if isinstance(result, list):
        return f"{len(result)} items"
    if isinstance(result, dict):
        return f"{len(result)} keys"
    return None
