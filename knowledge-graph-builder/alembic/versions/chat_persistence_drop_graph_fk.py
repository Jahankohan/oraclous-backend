"""Drop chat_conversations.graph_id FK to stale knowledge_graphs table

Revision ID: chat_persistence_drop_graph_fk
Revises: chat_persistence
Create Date: 2026-05-15 12:00:00.000000

Hotfix for STORY-031: the chat_persistence migration created a FK
from chat_conversations.graph_id to knowledge_graphs.id, but
STORY-025 had already moved graph identity to Neo4j — the Postgres
``knowledge_graphs`` table is no longer maintained. Every chat
turn for a graph created after STORY-025 failed with
ForeignKeyViolationError on the chat_conversations insert.

Fix:
- Drop the FK constraint. ``graph_id`` becomes a soft UUID
  reference; Neo4j is the source of truth for graph identity.
- Orphans become possible at the DB level if Neo4j graphs are
  deleted without cascading to Postgres. Acceptable trade-off for
  v1; a follow-up task (TASK-XXX) will move chat persistence to
  Neo4j under ``:__Chat__`` to eliminate the dangling reference
  entirely (Option 3 from the bug report).
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "chat_persistence_drop_graph_fk"
down_revision = "chat_persistence"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint(
        "chat_conversations_graph_id_fkey",
        "chat_conversations",
        type_="foreignkey",
    )


def downgrade():
    # Re-create the FK to restore the chat_persistence original shape.
    # Note: this will fail if any chat_conversations row has a graph_id
    # that doesn't exist in knowledge_graphs — which is exactly the
    # situation that motivated the drop. A real rollback requires
    # cleaning orphans or backfilling knowledge_graphs first.
    op.create_foreign_key(
        "chat_conversations_graph_id_fkey",
        source_table="chat_conversations",
        referent_table="knowledge_graphs",
        local_cols=["graph_id"],
        remote_cols=["id"],
        ondelete="CASCADE",
    )


# Re-export to satisfy linters that don't recognize ``sa``/UUID as used
# (they aren't directly used in this revision, but keeping the imports
# documented makes future edits cheaper).
_ = (sa, UUID)
