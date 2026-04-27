"""add connector tables: connectors, connector_sync_logs, webhook_events

Revision ID: add_connector_tables
Revises: add_community_detection_columns
Create Date: 2026-04-08 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision = 'add_connector_tables'
down_revision = 'add_community_detection_columns'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'connectors',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('graph_id', sa.Text(), nullable=False),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('connector_type', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False, server_default='active'),
        sa.Column('config', JSONB(), nullable=False),
        sa.Column('schedule', sa.Text(), nullable=True),
        sa.Column('last_synced_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_sync_cursor', JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    op.create_index('idx_connectors_graph_id', 'connectors', ['graph_id'])
    op.create_index('idx_connectors_user_id', 'connectors', ['user_id'])

    op.create_table(
        'connector_sync_logs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('connector_id', UUID(as_uuid=True), sa.ForeignKey('connectors.id', ondelete='CASCADE'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.Text(), nullable=True),
        sa.Column('items_processed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('entities_extracted', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', JSONB(), nullable=True),
    )
    op.create_index('idx_sync_logs_connector_id', 'connector_sync_logs', ['connector_id'])

    op.create_table(
        'webhook_events',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('connector_id', UUID(as_uuid=True), sa.ForeignKey('connectors.id', ondelete='CASCADE'), nullable=False),
        sa.Column('event_type', sa.Text(), nullable=True),
        sa.Column('payload_hash', sa.Text(), nullable=False),
        sa.Column('payload', JSONB(), nullable=False),
        sa.Column('received_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.Text(), nullable=False, server_default='pending'),
        sa.Column('error_message', sa.Text(), nullable=True),
    )
    op.create_unique_constraint('uq_webhook_dedup', 'webhook_events', ['connector_id', 'payload_hash'])
    op.create_index('idx_webhook_events_connector_id', 'webhook_events', ['connector_id'])
    op.create_index('idx_webhook_events_status', 'webhook_events', ['status'])


def downgrade():
    op.drop_table('webhook_events')
    op.drop_table('connector_sync_logs')
    op.drop_table('connectors')
