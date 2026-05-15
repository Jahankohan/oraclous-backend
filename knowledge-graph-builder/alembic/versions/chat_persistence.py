"""chat persistence: conversations, messages, tool calls, access log, RLS (STORY-031)

Revision ID: chat_persistence
Revises: add_blob_cas_table
Create Date: 2026-05-15 00:00:00.000000

Postgres source of truth for chat conversations, messages, per-tool
audit, feedback, and access logging (ADR-020). The Neo4j semantic
projection under ``:__Chat__`` is added by TASK-106.

RLS policies on ``chat_conversations`` and ``chat_messages`` enforce
per-user isolation: chat endpoints set ``app.current_user_id`` on
their session before any query. See ``get_chat_db`` in
``app/api/dependencies.py``.

Notes:

* The pre-STORY-031 ``app/models/chat.py`` defined ``ChatSession`` and a
  flat ``ChatMessage`` but no migration was ever generated for them, so
  this revision creates the tables fresh under the new names. No data
  loss possible — the prior tables never existed in any database.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "chat_persistence"
down_revision = "add_blob_cas_table"
branch_labels = None
depends_on = None


def upgrade():
    # ------------------------------------------------------------------ #
    # chat_conversations
    # ------------------------------------------------------------------ #
    op.create_table(
        "chat_conversations",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "graph_id",
            UUID(as_uuid=True),
            sa.ForeignKey("knowledge_graphs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("agent_id", UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "last_message_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_chat_conversations_user_graph_last_msg",
        "chat_conversations",
        ["user_id", "graph_id", sa.text("last_message_at DESC")],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "idx_chat_conversations_graph_agent_last_msg",
        "chat_conversations",
        ["graph_id", "agent_id", sa.text("last_message_at DESC")],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    # ------------------------------------------------------------------ #
    # chat_messages
    # ------------------------------------------------------------------ #
    op.create_table(
        "chat_messages",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "conversation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("chat_conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        # Audit metadata (nullable; assistant turns only)
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("provider", sa.Text(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=True),
        sa.Column("reasoning_mode", sa.Text(), nullable=True),
        sa.Column("retriever_used", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "cancelled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("sources", JSONB(), nullable=True),
        # Per-turn feedback
        sa.Column("feedback_rating", sa.SmallInteger(), nullable=True),
        sa.Column("feedback_comment", sa.Text(), nullable=True),
        sa.Column("feedback_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="chat_messages_role_check",
        ),
        sa.CheckConstraint(
            "feedback_rating IS NULL OR feedback_rating IN (-1, 1)",
            name="chat_messages_feedback_rating_check",
        ),
    )
    op.create_index(
        "idx_chat_messages_conversation_created",
        "chat_messages",
        ["conversation_id", "created_at"],
    )
    op.create_index(
        "idx_chat_messages_assistant_created",
        "chat_messages",
        ["conversation_id", "role", "created_at"],
        postgresql_where=sa.text("role = 'assistant'"),
    )

    # ------------------------------------------------------------------ #
    # chat_message_tool_calls
    # ------------------------------------------------------------------ #
    op.create_table(
        "chat_message_tool_calls",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "message_id",
            UUID(as_uuid=True),
            sa.ForeignKey("chat_messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sequence_index", sa.SmallInteger(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("args_json", JSONB(), nullable=False),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("result_compressed", sa.LargeBinary(), nullable=True),
        sa.Column("result_compression", sa.Text(), nullable=True),
        sa.Column("result_uncompressed_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column(
            "result_truncated",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_chat_tool_calls_message_sequence",
        "chat_message_tool_calls",
        ["message_id", "sequence_index"],
    )
    op.create_index(
        "idx_chat_tool_calls_tool_started",
        "chat_message_tool_calls",
        ["tool_name", sa.text("started_at DESC")],
    )

    # ------------------------------------------------------------------ #
    # chat_access_log
    # ------------------------------------------------------------------ #
    op.create_table(
        "chat_access_log",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("user_id", UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=True),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column(
            "accessed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_chat_access_log_user_accessed",
        "chat_access_log",
        ["user_id", sa.text("accessed_at DESC")],
    )
    op.create_index(
        "idx_chat_access_log_conversation_accessed",
        "chat_access_log",
        ["conversation_id", sa.text("accessed_at DESC")],
    )

    # ------------------------------------------------------------------ #
    # Row-Level Security policies — defense-in-depth (ADR-020 / TASK-107).
    #
    # The application MUST set app.current_user_id via set_config(...)
    # before any query against these tables. If the GUC is unset or
    # mismatched, RLS returns zero rows.
    #
    # Policies use FORCE so the table owner is also subject to them
    # (without FORCE, a privileged role bypasses RLS). The owning role
    # is the app role configured in POSTGRES_URL.
    # ------------------------------------------------------------------ #
    op.execute("ALTER TABLE chat_conversations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE chat_conversations FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY chat_conversations_user_isolation
        ON chat_conversations
        USING (user_id = current_setting('app.current_user_id', true)::uuid)
        WITH CHECK (user_id = current_setting('app.current_user_id', true)::uuid)
        """
    )

    op.execute("ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE chat_messages FORCE ROW LEVEL SECURITY")
    # Messages inherit isolation via the conversation FK.
    op.execute(
        """
        CREATE POLICY chat_messages_user_isolation
        ON chat_messages
        USING (
            conversation_id IN (
                SELECT id FROM chat_conversations
                WHERE user_id = current_setting('app.current_user_id', true)::uuid
            )
        )
        WITH CHECK (
            conversation_id IN (
                SELECT id FROM chat_conversations
                WHERE user_id = current_setting('app.current_user_id', true)::uuid
            )
        )
        """
    )

    # Tool calls and access log are not RLS-protected at the table level —
    # tool_calls inherit safety via the chat_messages RLS chain when joined,
    # and access_log is admin-read-only with no per-user read endpoint.
    # If a future endpoint exposes access_log per-user, add a policy here.


def downgrade():
    op.execute("DROP POLICY IF EXISTS chat_messages_user_isolation ON chat_messages")
    op.execute(
        "DROP POLICY IF EXISTS chat_conversations_user_isolation ON chat_conversations"
    )
    op.execute("ALTER TABLE chat_messages DISABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE chat_conversations DISABLE ROW LEVEL SECURITY")

    op.drop_index(
        "idx_chat_access_log_conversation_accessed",
        table_name="chat_access_log",
    )
    op.drop_index("idx_chat_access_log_user_accessed", table_name="chat_access_log")
    op.drop_table("chat_access_log")

    op.drop_index(
        "idx_chat_tool_calls_tool_started",
        table_name="chat_message_tool_calls",
    )
    op.drop_index(
        "idx_chat_tool_calls_message_sequence",
        table_name="chat_message_tool_calls",
    )
    op.drop_table("chat_message_tool_calls")

    op.drop_index("idx_chat_messages_assistant_created", table_name="chat_messages")
    op.drop_index("idx_chat_messages_conversation_created", table_name="chat_messages")
    op.drop_table("chat_messages")

    op.drop_index(
        "idx_chat_conversations_graph_agent_last_msg",
        table_name="chat_conversations",
    )
    op.drop_index(
        "idx_chat_conversations_user_graph_last_msg",
        table_name="chat_conversations",
    )
    op.drop_table("chat_conversations")
