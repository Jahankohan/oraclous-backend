"""Unit tests for the chat persistence models (STORY-031 / TASK-102).

These tests verify the model module imports cleanly, registers each
model on the shared SQLAlchemy ``Base.metadata``, and exposes the
expected table and column shape. They run without a Postgres connection
— a real-database roundtrip is in
``tests/integration/test_chat_rls_integration.py``.
"""

from __future__ import annotations

import pytest

from app.core.database import Base
from app.models.chat import (
    ChatAccessLog,
    ChatConversation,
    ChatMessage,
    ChatMessageToolCall,
)


@pytest.mark.unit
class TestChatModelsRegistration:
    def test_all_four_tables_registered(self):
        """The four chat tables must appear on the shared metadata."""
        tables = Base.metadata.tables
        assert "chat_conversations" in tables
        assert "chat_messages" in tables
        assert "chat_message_tool_calls" in tables
        assert "chat_access_log" in tables

    def test_no_stub_chat_sessions_table(self):
        """The pre-STORY-031 ``chat_sessions`` stub model must be gone."""
        assert "chat_sessions" not in Base.metadata.tables

    def test_conversation_columns(self):
        cols = {c.name for c in ChatConversation.__table__.columns}
        assert cols == {
            "id",
            "user_id",
            "graph_id",
            "agent_id",
            "title",
            "created_at",
            "last_message_at",
            "deleted_at",
        }

    def test_message_columns(self):
        cols = {c.name for c in ChatMessage.__table__.columns}
        assert "id" in cols
        assert "conversation_id" in cols
        # Forward-compat session seam
        assert "session_id" in cols
        # Required content + role
        assert "role" in cols
        assert "content" in cols
        # Audit metadata
        for col in [
            "model",
            "provider",
            "prompt_tokens",
            "completion_tokens",
            "latency_ms",
            "cost_usd",
            "reasoning_mode",
            "retriever_used",
            "error",
            "cancelled",
            "sources",
        ]:
            assert col in cols, f"missing audit column: {col}"
        # Feedback
        for col in ["feedback_rating", "feedback_comment", "feedback_at"]:
            assert col in cols, f"missing feedback column: {col}"

    def test_message_session_id_is_nullable(self):
        """Session id is reserved nullable until v2 (per STORY-031)."""
        col = ChatMessage.__table__.c.session_id
        assert col.nullable is True

    def test_tool_call_columns_have_compression_seam(self):
        cols = {c.name for c in ChatMessageToolCall.__table__.columns}
        for col in [
            "result_compressed",
            "result_compression",
            "result_uncompressed_size_bytes",
            "result_truncated",
        ]:
            assert col in cols, f"missing compression column: {col}"

    def test_role_check_constraint_present(self):
        """The role-validity check constraint must be on the message table."""
        constraint_names = {c.name for c in ChatMessage.__table__.constraints if c.name}
        assert "chat_messages_role_check" in constraint_names

    def test_feedback_rating_check_constraint_present(self):
        constraint_names = {c.name for c in ChatMessage.__table__.constraints if c.name}
        assert "chat_messages_feedback_rating_check" in constraint_names

    def test_conversation_to_message_cascade(self):
        """Deleting a conversation cascades to its messages (orphan-free)."""
        fk = next(iter(ChatMessage.__table__.foreign_keys))
        assert fk.column.table.name == "chat_conversations"
        assert fk.ondelete == "CASCADE"

    def test_message_to_tool_call_cascade(self):
        fk = next(iter(ChatMessageToolCall.__table__.foreign_keys))
        assert fk.column.table.name == "chat_messages"
        assert fk.ondelete == "CASCADE"

    def test_access_log_no_user_fk(self):
        """``user_id`` on access log is a plain UUID — users live in the auth service."""
        col = ChatAccessLog.__table__.c.user_id
        assert col.foreign_keys == set()

    def test_conversation_no_user_fk(self):
        """``user_id`` on conversation is a plain UUID — see above."""
        col = ChatConversation.__table__.c.user_id
        assert col.foreign_keys == set()

    def test_conversation_no_agent_fk(self):
        """``agent_id`` is a plain UUID — agents live in Neo4j, not Postgres."""
        col = ChatConversation.__table__.c.agent_id
        assert col.foreign_keys == set()
        assert col.nullable is True
