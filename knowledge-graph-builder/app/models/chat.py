"""Chat persistence models (STORY-031, ADR-020).

These tables are the Postgres source of truth for chat conversations,
messages, tool-call audit, feedback, and access logging. A separate
Celery task projects a lightweight semantic shadow into Neo4j under the
reserved ``:__Chat__`` namespace label (TASK-106) — Postgres is always
authoritative; Neo4j projection failures never block the user write.

Row-level security policies on ``chat_conversations`` and
``chat_messages`` enforce per-user isolation: every connection that
serves a chat endpoint sets ``app.current_user_id`` via
``get_chat_db`` (see app.api.dependencies). This is defense-in-depth on
top of the explicit per-query ``user_id`` filter.

Sessions are explicitly deferred to v2 (see STORY-031 non-goals). The
``chat_messages.session_id`` column is reserved nullable from day one
so a future migration can begin populating it without a data backfill.
"""

import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class ChatConversation(Base):
    """A user-facing chat thread.

    One per "New chat" click — the unit the user sees in the conversation
    list. ``agent_id`` is nullable for graph chat (no agent bound).
    """

    __tablename__ = "chat_conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # No FK on user_id — users live in the auth service, not Postgres.
    user_id = Column(UUID(as_uuid=True), nullable=False)
    # No FK on graph_id either: STORY-025 moved graph identity to Neo4j
    # (the Postgres ``knowledge_graphs`` table is no longer maintained).
    # See alembic/versions/chat_persistence_drop_graph_fk.py. A future
    # task should move chat persistence to Neo4j under :__Chat__ to
    # eliminate the soft reference entirely.
    graph_id = Column(UUID(as_uuid=True), nullable=False)
    # No FK on agent_id — agents live in Neo4j (the :Agent:__Platform__ nodes).
    agent_id = Column(UUID(as_uuid=True), nullable=True)
    title = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_message_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    # Soft-delete; a 30d sweeper Celery task hard-deletes (TASK-104).
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    messages = relationship(
        "ChatMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index(
            "idx_chat_conversations_user_graph_last_msg",
            "user_id",
            "graph_id",
            last_message_at.desc(),
            postgresql_where=deleted_at.is_(None),
        ),
        Index(
            "idx_chat_conversations_graph_agent_last_msg",
            "graph_id",
            "agent_id",
            last_message_at.desc(),
            postgresql_where=deleted_at.is_(None),
        ),
    )


class ChatMessage(Base):
    """One chat turn (user or assistant).

    Audit metadata columns are populated on assistant turns only.
    Feedback columns are populated when the user rates the turn.
    ``session_id`` is reserved nullable for the v2 session feature.
    """

    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Forward-compat seam for v2 sessions (STORY-031 deferred decision).
    session_id = Column(UUID(as_uuid=True), nullable=True)
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Audit metadata — populated for assistant turns only.
    model = Column(Text, nullable=True)
    provider = Column(Text, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    cost_usd = Column(Numeric(10, 6), nullable=True)
    reasoning_mode = Column(Text, nullable=True)
    retriever_used = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    cancelled = Column(Boolean, nullable=False, server_default="false", default=False)
    sources = Column(JSONB, nullable=True)

    # Per-turn feedback (👍 / 👎 + optional comment) — TASK-104 endpoint.
    feedback_rating = Column(SmallInteger, nullable=True)
    feedback_comment = Column(Text, nullable=True)
    feedback_at = Column(DateTime(timezone=True), nullable=True)

    conversation = relationship("ChatConversation", back_populates="messages")
    tool_calls = relationship(
        "ChatMessageToolCall",
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessageToolCall.sequence_index",
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="chat_messages_role_check",
        ),
        CheckConstraint(
            "feedback_rating IS NULL OR feedback_rating IN (-1, 1)",
            name="chat_messages_feedback_rating_check",
        ),
        Index(
            "idx_chat_messages_conversation_created",
            "conversation_id",
            "created_at",
        ),
        Index(
            "idx_chat_messages_assistant_created",
            "conversation_id",
            "role",
            "created_at",
            postgresql_where=text("role = 'assistant'"),
        ),
    )


class ChatMessageToolCall(Base):
    """Per-tool invocation audit row.

    Stores the full result blob zstd-compressed. The 5 MB compressed cap
    (and 50 MB raw cap) is enforced at the service layer in TASK-103;
    over-cap calls are persisted with ``result_truncated=true`` and a
    minimal placeholder in ``result_compressed``.
    """

    __tablename__ = "chat_message_tool_calls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    sequence_index = Column(SmallInteger, nullable=False)
    tool_name = Column(Text, nullable=False)
    args_json = Column(JSONB, nullable=False)
    result_summary = Column(Text, nullable=True)
    result_compressed = Column(LargeBinary, nullable=True)
    result_compression = Column(Text, nullable=True)
    result_uncompressed_size_bytes = Column(BigInteger, nullable=True)
    result_truncated = Column(
        Boolean, nullable=False, server_default="false", default=False
    )
    latency_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    started_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    message = relationship("ChatMessage", back_populates="tool_calls")

    __table_args__ = (
        Index(
            "idx_chat_tool_calls_message_sequence",
            "message_id",
            "sequence_index",
        ),
        Index(
            "idx_chat_tool_calls_tool_started",
            "tool_name",
            started_at.desc(),
        ),
    )


class ChatAccessLog(Base):
    """Audit trail of who accessed which chat resource.

    Populated by an async Celery task triggered from the read endpoints
    in TASK-104. Sampled (1/N) so it never blocks the hot path.
    """

    __tablename__ = "chat_access_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    # Null if the access was a list call (no specific conversation).
    conversation_id = Column(UUID(as_uuid=True), nullable=True)
    endpoint = Column(Text, nullable=False)
    accessed_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index(
            "idx_chat_access_log_user_accessed",
            "user_id",
            accessed_at.desc(),
        ),
        Index(
            "idx_chat_access_log_conversation_accessed",
            "conversation_id",
            accessed_at.desc(),
        ),
    )
