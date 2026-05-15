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
from sqlalchemy import asc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import (
    ChatConversation,
    ChatMessage,
    ChatMessageToolCall,
)
from app.services.llm_pricing import estimate_cost_usd

logger = logging.getLogger(__name__)

# Token estimator — lazy-loaded so tests that don't use it don't pay the
# import cost. ``cl100k_base`` is a sensible default approximation for
# both OpenAI and Anthropic models; it overestimates slightly for
# Claude (which uses a different tokenizer) but the cap is a soft
# guardrail, not a hard contract.
_TIKTOKEN_ENCODER = None


def _estimate_tokens(text: str) -> int:
    """Rough token count for context budgeting. Best-effort."""
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        try:
            import tiktoken

            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover  — tiktoken always installed
            _TIKTOKEN_ENCODER = False
    if _TIKTOKEN_ENCODER is False:
        # Char/4 ≈ tokens for English. Coarse but acceptable as a
        # last-resort fallback.
        return max(1, len(text) // 4)
    return len(_TIKTOKEN_ENCODER.encode(text))


# Caps for tool result persistence (ADR-020 / STORY-031 decisions table).
MAX_RAW_BYTES = 50 * 1024 * 1024  # 50 MB pre-compression
MAX_COMPRESSED_BYTES = 5 * 1024 * 1024  # 5 MB compressed cap

# Hard cap on decompression output for adversarial zstd payloads (TASK-107).
# A compressed blob that decompresses past this cap is rejected — the
# write-path cap (MAX_RAW_BYTES) is the natural ceiling; this is the
# safety net on the read path so a corrupted/adversarial row can't
# allocate gigabytes of memory.
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024  # 64 MB

_ZSTD_LEVEL = 3
_compressor = zstd.ZstdCompressor(level=_ZSTD_LEVEL)


class CompressionLimitExceeded(Exception):
    """Decompressed payload exceeded MAX_DECOMPRESSED_BYTES (TASK-107)."""


def decompress_result(
    compressed: bytes | None,
    compression: str | None,
) -> bytes | None:
    """Decompress a tool-call result blob, with a hard size cap.

    Reader-side counterpart to ``_compress_result``. Future readers
    (export endpoint, future memory features) MUST use this rather
    than calling ``zstd.ZstdDecompressor().decompress()`` directly —
    the cap is the only defense against an adversarial blob whose
    decompression would balloon memory.

    Raises ``CompressionLimitExceeded`` if the payload would exceed
    ``MAX_DECOMPRESSED_BYTES`` (default 64 MB).
    """
    if compressed is None:
        return None
    if compression != "zstd":
        # Unknown / no compression — return as-is.
        return compressed
    # Use streaming decompression so we can short-circuit before
    # allocating the full output buffer.
    dctx = zstd.ZstdDecompressor()
    out = bytearray()
    with dctx.stream_reader(compressed) as reader:
        while True:
            chunk = reader.read(65536)
            if not chunk:
                break
            out.extend(chunk)
            if len(out) > MAX_DECOMPRESSED_BYTES:
                raise CompressionLimitExceeded(
                    f"decompressed payload exceeded "
                    f"{MAX_DECOMPRESSED_BYTES} bytes; refusing to allocate"
                )
    return bytes(out)


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

    async def load_context(
        self,
        db: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        max_turns: int = 20,
        max_tokens: int = 8000,
    ) -> list[dict[str, str]]:
        """Load prior turns to inject as conversation context for the agent.

        Returns ``[{"role": "user|assistant", "content": "..."}]`` ordered
        oldest → newest. Trimmed from the oldest end until both caps hold.

        Excludes:
        * ``cancelled = TRUE`` assistant turns — partial garbage is worse
          than missing context.
        * Empty assistant turns (error cases where ``content = ''``).
        * System role turns — those are owned by the agent definition,
          not by the conversation.

        The current in-flight user message MUST NOT be in the conversation
        yet when this is called — callers should invoke ``load_context``
        BEFORE ``write_user_message``.
        """
        stmt = (
            select(ChatMessage)
            .where(
                ChatMessage.conversation_id == conversation_id,
                ChatMessage.role.in_(("user", "assistant")),
            )
            .order_by(asc(ChatMessage.created_at))
        )
        rows = list((await db.execute(stmt)).scalars().all())

        # Filter out unusable assistant turns.
        cleaned: list[dict[str, str]] = []
        for m in rows:
            if m.role == "assistant" and (m.cancelled or not m.content):
                continue
            cleaned.append({"role": m.role, "content": m.content})

        # Hard turn cap from the newest end.
        if len(cleaned) > max_turns:
            cleaned = cleaned[-max_turns:]

        # Token cap — drop oldest entries until total estimated tokens
        # are <= max_tokens.
        total = sum(_estimate_tokens(m["content"]) for m in cleaned)
        while cleaned and total > max_tokens:
            removed = cleaned.pop(0)
            total -= _estimate_tokens(removed["content"])

        return cleaned

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
