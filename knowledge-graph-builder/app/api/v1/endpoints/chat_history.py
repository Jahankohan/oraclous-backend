"""Chat history endpoints (STORY-031 / TASK-104).

Read-side surface for the persisted chat data written by TASK-103.
Companion to the deprecated ``GET /graphs/{graph_id}/chat/history``
flat-list endpoint in ``graphs.py:454-484`` — that one continues to
return ``[]`` for one release cycle so existing clients don't 410.

Endpoint table:

| Method | Path                                          | Owner action |
|--------|-----------------------------------------------|--------------|
| GET    | /graphs/{gid}/chat/conversations              | list         |
| POST   | /graphs/{gid}/chat/conversations              | create       |
| GET    | /chat/conversations/{cid}/messages            | read         |
| PATCH  | /chat/conversations/{cid}                     | rename       |
| DELETE | /chat/conversations/{cid}                     | soft-delete  |
| POST   | /chat/messages/{mid}/feedback                 | feedback set |
| DELETE | /chat/messages/{mid}/feedback                 | feedback clr |
| GET    | /me/chat/export                               | regulatory   |

Security model (ADR-020):

* Every endpoint depends on ``get_chat_db`` which sets the
  ``app.current_user_id`` GUC. RLS policies on chat tables enforce
  per-user isolation as defense-in-depth.
* Every read additionally adds ``user_id = current_user_id`` to its
  query as the primary filter — explicit > implicit.
* Cross-user reads return 404 (never 403) to avoid id enumeration.
* When ``graph_id`` is in the path, ``verify_graph_access(..., 'read')``
  runs against the ReBAC layer in addition to the user filter.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import asc, delete, desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import (
    get_chat_db,
    get_current_user_id,
    verify_graph_access,
)
from app.core.logging import get_logger
from app.models.chat import (
    ChatConversation,
    ChatMessage,
    ChatMessageToolCall,
)
from app.schemas.chat_schemas import (
    ConversationsListResponse,
    ConversationSummary,
    CreateConversationRequest,
    CreateConversationResponse,
    FeedbackRequest,
    MessagesListResponse,
    MessageWithMetadata,
    PatchConversationRequest,
    ToolCallEntry,
)

router = APIRouter()
logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers
# ──────────────────────────────────────────────────────────────────────────── #


def _conv_to_summary(conv: ChatConversation) -> ConversationSummary:
    return ConversationSummary(
        id=str(conv.id),
        agent_id=str(conv.agent_id) if conv.agent_id else None,
        title=conv.title,
        created_at=conv.created_at,
        last_message_at=conv.last_message_at,
    )


def _msg_to_response(
    msg: ChatMessage,
    tool_calls: list[ChatMessageToolCall],
) -> MessageWithMetadata:
    return MessageWithMetadata(
        id=str(msg.id),
        role=msg.role,
        content=msg.content,
        created_at=msg.created_at,
        model=msg.model,
        provider=msg.provider,
        prompt_tokens=msg.prompt_tokens,
        completion_tokens=msg.completion_tokens,
        latency_ms=msg.latency_ms,
        cost_usd=str(msg.cost_usd) if msg.cost_usd is not None else None,
        reasoning_mode=msg.reasoning_mode,
        retriever_used=msg.retriever_used,
        error=msg.error,
        cancelled=msg.cancelled,
        sources=msg.sources,
        feedback_rating=msg.feedback_rating,
        feedback_comment=msg.feedback_comment,
        feedback_at=msg.feedback_at,
        tool_calls=[
            ToolCallEntry(
                id=str(tc.id),
                sequence_index=tc.sequence_index,
                tool_name=tc.tool_name,
                args_json=tc.args_json,
                result_summary=tc.result_summary,
                result_truncated=tc.result_truncated,
                latency_ms=tc.latency_ms,
                error=tc.error,
            )
            for tc in tool_calls
        ],
    )


def _parse_uuid_or_404(value: str, label: str) -> uuid.UUID:
    """Convert a path/query string to UUID or raise 404.

    404 (not 422) is intentional — keeps the API consistent with the
    cross-user 404 strategy so id enumeration cannot distinguish
    'malformed' from 'not yours'.
    """
    try:
        return uuid.UUID(value)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown {label}"
        ) from None


async def _load_owned_conversation(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    include_deleted: bool = False,
) -> ChatConversation:
    """Return the conversation if the caller owns it, else 404."""
    where_clauses = [
        ChatConversation.id == conversation_id,
        ChatConversation.user_id == user_id,
    ]
    if not include_deleted:
        where_clauses.append(ChatConversation.deleted_at.is_(None))
    stmt = select(ChatConversation).where(*where_clauses)
    conv = (await db.execute(stmt)).scalar_one_or_none()
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    return conv


# ──────────────────────────────────────────────────────────────────────────── #
# Endpoints
# ──────────────────────────────────────────────────────────────────────────── #


@router.get(
    "/graphs/{graph_id}/chat/conversations",
    response_model=ConversationsListResponse,
    summary="List the current user's conversations in this graph",
    responses={403: {"description": "Access denied"}},
)
async def list_conversations(
    graph_id: str,
    agent_id: str | None = Query(
        default=None, description="Filter to a single agent's conversations."
    ),
    limit: int = Query(default=20, ge=1, le=100),
    before: str | None = Query(
        default=None,
        description="Cursor — an ISO8601 timestamp of the last item from the "
        "previous page. Items strictly older than this are returned.",
    ),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_chat_db),
) -> ConversationsListResponse:
    await verify_graph_access(graph_id, "read", user_id)

    user_uuid = uuid.UUID(user_id)
    graph_uuid = uuid.UUID(graph_id)

    stmt = (
        select(ChatConversation)
        .where(
            ChatConversation.user_id == user_uuid,
            ChatConversation.graph_id == graph_uuid,
            ChatConversation.deleted_at.is_(None),
        )
        .order_by(desc(ChatConversation.last_message_at))
        .limit(limit + 1)  # +1 to detect another page
    )

    if agent_id is not None:
        try:
            stmt = stmt.where(ChatConversation.agent_id == uuid.UUID(agent_id))
        except (ValueError, TypeError):
            # Unknown agent id → return an empty page.
            return ConversationsListResponse(items=[], next_cursor=None)

    if before is not None:
        try:
            before_dt = datetime.fromisoformat(before)
            stmt = stmt.where(ChatConversation.last_message_at < before_dt)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid before cursor: {exc}",
            ) from None

    rows = list((await db.execute(stmt)).scalars().all())

    has_more = len(rows) > limit
    page = rows[:limit]
    next_cursor: str | None = None
    if has_more:
        next_cursor = page[-1].last_message_at.isoformat()

    return ConversationsListResponse(
        items=[_conv_to_summary(c) for c in page],
        next_cursor=next_cursor,
    )


@router.post(
    "/graphs/{graph_id}/chat/conversations",
    response_model=CreateConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an empty conversation explicitly",
    responses={403: {"description": "Access denied"}},
)
async def create_conversation(
    graph_id: str,
    body: CreateConversationRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_chat_db),
) -> CreateConversationResponse:
    """Create a fresh conversation up-front.

    Most clients won't call this — the chat handlers in TASK-103
    auto-create on the first turn. This endpoint is for clients that
    want to allocate the id before sending the first message (e.g.,
    for FE optimistic-UI flows).
    """
    await verify_graph_access(graph_id, "read", user_id)

    agent_uuid = None
    if body.agent_id is not None:
        try:
            agent_uuid = uuid.UUID(body.agent_id)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid agent_id"
            ) from None

    conv = ChatConversation(
        user_id=uuid.UUID(user_id),
        graph_id=uuid.UUID(graph_id),
        agent_id=agent_uuid,
        title=(body.title or "New chat")[:200],
    )
    db.add(conv)
    await db.flush()
    await db.commit()
    return CreateConversationResponse(id=str(conv.id), title=conv.title)


@router.get(
    "/chat/conversations/{conversation_id}/messages",
    response_model=MessagesListResponse,
    summary="Return the messages for a conversation",
    responses={404: {"description": "Conversation not found or not yours"}},
)
async def list_messages(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    before: str | None = Query(
        default=None,
        description="ISO8601 timestamp; returns messages strictly older.",
    ),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_chat_db),
) -> MessagesListResponse:
    cid = _parse_uuid_or_404(conversation_id, "conversation")
    await _load_owned_conversation(db, conversation_id=cid, user_id=uuid.UUID(user_id))

    msg_stmt = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == cid)
        .order_by(asc(ChatMessage.created_at))
        .limit(limit + 1)
    )
    if before is not None:
        try:
            before_dt = datetime.fromisoformat(before)
            msg_stmt = msg_stmt.where(ChatMessage.created_at < before_dt)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid before cursor: {exc}",
            ) from None

    rows: list[ChatMessage] = list((await db.execute(msg_stmt)).scalars().all())
    has_more = len(rows) > limit
    page = rows[:limit]

    # Bulk-load tool calls for this page.
    msg_ids = [m.id for m in page]
    tool_calls: list[ChatMessageToolCall] = []
    if msg_ids:
        tc_stmt = (
            select(ChatMessageToolCall)
            .where(ChatMessageToolCall.message_id.in_(msg_ids))
            .order_by(
                ChatMessageToolCall.message_id, ChatMessageToolCall.sequence_index
            )
        )
        tool_calls = list((await db.execute(tc_stmt)).scalars().all())

    by_message: dict[uuid.UUID, list[ChatMessageToolCall]] = {}
    for tc in tool_calls:
        by_message.setdefault(tc.message_id, []).append(tc)

    items = [_msg_to_response(m, by_message.get(m.id, [])) for m in page]
    next_cursor = page[-1].created_at.isoformat() if has_more else None
    return MessagesListResponse(items=items, next_cursor=next_cursor)


@router.patch(
    "/chat/conversations/{conversation_id}",
    response_model=ConversationSummary,
    summary="Rename a conversation",
    responses={404: {"description": "Conversation not found or not yours"}},
)
async def patch_conversation(
    conversation_id: str,
    body: PatchConversationRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_chat_db),
) -> ConversationSummary:
    cid = _parse_uuid_or_404(conversation_id, "conversation")
    conv = await _load_owned_conversation(
        db, conversation_id=cid, user_id=uuid.UUID(user_id)
    )
    if body.title is not None:
        conv.title = body.title[:200]
        await db.flush()
        await db.commit()
    return _conv_to_summary(conv)


@router.delete(
    "/chat/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a conversation (30d TTL then hard-delete)",
    responses={404: {"description": "Conversation not found or not yours"}},
)
async def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_chat_db),
):
    cid = _parse_uuid_or_404(conversation_id, "conversation")
    user_uuid = uuid.UUID(user_id)
    # Confirm ownership before mutating — _load_owned_conversation raises 404.
    await _load_owned_conversation(db, conversation_id=cid, user_id=user_uuid)
    stmt = (
        update(ChatConversation)
        .where(
            ChatConversation.id == cid,
            ChatConversation.user_id == user_uuid,
            ChatConversation.deleted_at.is_(None),
        )
        .values(deleted_at=datetime.utcnow())
    )
    await db.execute(stmt)
    await db.commit()
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)


# ──────────────────────────────────────────────────────────────────────────── #
# Feedback
# ──────────────────────────────────────────────────────────────────────────── #


def _check_feedback_rating(rating: int) -> int:
    if rating not in (-1, 1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="rating must be +1 or -1",
        )
    return rating


@router.post(
    "/chat/messages/{message_id}/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Submit feedback for an assistant message",
    responses={
        400: {"description": "Invalid rating"},
        404: {"description": "Message not found or not yours"},
    },
)
async def set_feedback(
    message_id: str,
    body: FeedbackRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_chat_db),
):
    mid = _parse_uuid_or_404(message_id, "message")
    rating = _check_feedback_rating(body.rating)

    # Verify the message belongs to a conversation the user owns via a join.
    user_uuid = uuid.UUID(user_id)
    stmt = (
        select(ChatMessage)
        .join(ChatConversation, ChatConversation.id == ChatMessage.conversation_id)
        .where(
            ChatMessage.id == mid,
            ChatConversation.user_id == user_uuid,
        )
    )
    msg = (await db.execute(stmt)).scalar_one_or_none()
    if msg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
        )

    msg.feedback_rating = rating
    msg.feedback_comment = body.comment
    msg.feedback_at = datetime.utcnow()
    await db.flush()
    await db.commit()
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)


@router.delete(
    "/chat/messages/{message_id}/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear feedback on a message",
    responses={404: {"description": "Message not found or not yours"}},
)
async def clear_feedback(
    message_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_chat_db),
):
    mid = _parse_uuid_or_404(message_id, "message")
    user_uuid = uuid.UUID(user_id)
    stmt = (
        select(ChatMessage)
        .join(ChatConversation, ChatConversation.id == ChatMessage.conversation_id)
        .where(
            ChatMessage.id == mid,
            ChatConversation.user_id == user_uuid,
        )
    )
    msg = (await db.execute(stmt)).scalar_one_or_none()
    if msg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
        )
    msg.feedback_rating = None
    msg.feedback_comment = None
    msg.feedback_at = None
    await db.flush()
    await db.commit()
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)


# ──────────────────────────────────────────────────────────────────────────── #
# Export (regulatory hygiene — STORY-031 ACs)
# ──────────────────────────────────────────────────────────────────────────── #


@router.get(
    "/me/chat/export",
    summary="Export the current user's chat history as JSON",
    response_class=StreamingResponse,
)
async def export_my_chats(
    include_raw: bool = Query(
        default=False,
        description="If true, include compressed tool-call result blobs "
        "(base64-encoded). Default is summaries only.",
    ),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_chat_db),
) -> StreamingResponse:
    """Streamed JSON dump of every conversation owned by the caller.

    ``include_raw=true`` adds the compressed tool result blobs. Without
    it, only summaries + truncation flags are returned — far smaller
    on the wire and sufficient for most regulatory exports.

    NOTE: this endpoint is intentionally simple — one conversation at
    a time, in created order. A future iteration can add format
    options (NDJSON, ZIP), but the v1 contract is a single JSON object.
    """
    user_uuid = uuid.UUID(user_id)
    conv_stmt = (
        select(ChatConversation)
        .where(ChatConversation.user_id == user_uuid)
        .order_by(asc(ChatConversation.created_at))
    )
    conversations = list((await db.execute(conv_stmt)).scalars().all())

    async def gen():
        import base64

        yield '{"conversations": ['
        first = True
        for conv in conversations:
            if not first:
                yield ","
            first = False
            msg_stmt = (
                select(ChatMessage)
                .where(ChatMessage.conversation_id == conv.id)
                .order_by(asc(ChatMessage.created_at))
            )
            msgs = list((await db.execute(msg_stmt)).scalars().all())
            tc_by_msg: dict[uuid.UUID, list[dict]] = {}
            if msgs:
                tc_stmt = (
                    select(ChatMessageToolCall)
                    .where(ChatMessageToolCall.message_id.in_([m.id for m in msgs]))
                    .order_by(
                        ChatMessageToolCall.message_id,
                        ChatMessageToolCall.sequence_index,
                    )
                )
                for tc in (await db.execute(tc_stmt)).scalars().all():
                    entry = {
                        "id": str(tc.id),
                        "sequence_index": tc.sequence_index,
                        "tool_name": tc.tool_name,
                        "args_json": tc.args_json,
                        "result_summary": tc.result_summary,
                        "result_truncated": tc.result_truncated,
                        "latency_ms": tc.latency_ms,
                        "error": tc.error,
                    }
                    if include_raw and tc.result_compressed is not None:
                        entry["result_compressed_base64"] = base64.b64encode(
                            tc.result_compressed
                        ).decode("ascii")
                        entry["result_compression"] = tc.result_compression
                    tc_by_msg.setdefault(tc.message_id, []).append(entry)

            payload = {
                "id": str(conv.id),
                "title": conv.title,
                "agent_id": str(conv.agent_id) if conv.agent_id else None,
                "created_at": conv.created_at.isoformat(),
                "deleted_at": (
                    conv.deleted_at.isoformat() if conv.deleted_at else None
                ),
                "messages": [
                    {
                        "id": str(m.id),
                        "role": m.role,
                        "content": m.content,
                        "created_at": m.created_at.isoformat(),
                        "model": m.model,
                        "provider": m.provider,
                        "prompt_tokens": m.prompt_tokens,
                        "completion_tokens": m.completion_tokens,
                        "latency_ms": m.latency_ms,
                        "cost_usd": (
                            str(m.cost_usd) if m.cost_usd is not None else None
                        ),
                        "reasoning_mode": m.reasoning_mode,
                        "retriever_used": m.retriever_used,
                        "error": m.error,
                        "cancelled": m.cancelled,
                        "sources": m.sources,
                        "feedback_rating": m.feedback_rating,
                        "feedback_comment": m.feedback_comment,
                        "tool_calls": tc_by_msg.get(m.id, []),
                    }
                    for m in msgs
                ],
            }
            yield json.dumps(payload)
        yield "]}"

    return StreamingResponse(
        gen(),
        media_type="application/json",
        headers={
            "Content-Disposition": "attachment; filename=oraclous-chat-export.json"
        },
    )


# Re-export ``delete`` to mirror SQLAlchemy import in tests if needed.
__all__ = ["router", "delete"]
