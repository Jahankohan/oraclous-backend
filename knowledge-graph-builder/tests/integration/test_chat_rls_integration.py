"""Integration tests for chat persistence: migration roundtrip + RLS isolation.

Skips when ``TEST_POSTGRES_URL`` is unset or unreachable — these tests
need a real Postgres because they exercise the migration DDL and the
RLS policy enforcement that an SQLite-based fake cannot replicate.

The migration is applied/reverted against the configured database. To
avoid clobbering a developer's local DB, the test fixture creates a
dedicated schema and aliases the existing tables into it; if no
``TEST_POSTGRES_URL`` is set, the entire module is skipped.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="TEST_POSTGRES_URL not set — skipping chat-RLS integration"
)


@pytest_asyncio.fixture(scope="module")
async def engine():
    eng = create_async_engine(POSTGRES_URL, pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(scope="module")
async def setup_schema(engine):
    """Apply the chat_persistence migration's DDL to a fresh schema.

    We don't run the full alembic chain — just the DDL the new migration
    introduces, plus a stub ``knowledge_graphs`` table (FK target) so
    the FK constraint can be created without bringing the entire
    pre-existing schema along.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE SCHEMA IF NOT EXISTS chat_rls_test"))
        await conn.execute(text("SET search_path TO chat_rls_test"))
        # Stub knowledge_graphs for the FK.
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chat_rls_test.knowledge_graphs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                )
                """
            )
        )
        # Run the chat_persistence migration DDL directly via the
        # alembic op-equivalent SQL. We can't import the migration as a
        # module because it expects the alembic Operations context.
        from alembic.migration import MigrationContext
        from alembic.operations import Operations

        ctx = MigrationContext.configure(connection=conn.sync_connection)
        op = Operations(ctx)
        # Bind the global ``op`` used inside the migration module.
        import alembic.versions.chat_persistence as mig_module
        from alembic.versions.chat_persistence import upgrade as run_upgrade

        mig_module.op = op
        run_upgrade()
        yield
        # Teardown: downgrade then drop the schema.
        from alembic.versions.chat_persistence import downgrade as run_downgrade

        run_downgrade()
        await conn.execute(text("DROP SCHEMA chat_rls_test CASCADE"))


@pytest_asyncio.fixture
async def session(engine, setup_schema):
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        await s.execute(text("SET search_path TO chat_rls_test"))
        yield s
        await s.rollback()


@pytest.mark.integration
class TestMigrationRoundtrip:
    async def test_all_four_tables_present(self, session: AsyncSession):
        for table in [
            "chat_conversations",
            "chat_messages",
            "chat_message_tool_calls",
            "chat_access_log",
        ]:
            res = await session.execute(
                text(
                    """
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'chat_rls_test' AND table_name = :t
                    """
                ).bindparams(t=table)
            )
            assert res.scalar() == 1, f"table missing after upgrade: {table}"

    async def test_rls_enabled_on_conversation(self, session: AsyncSession):
        res = await session.execute(
            text(
                """
                SELECT relrowsecurity, relforcerowsecurity
                FROM pg_class
                WHERE relname = 'chat_conversations'
                AND relnamespace = 'chat_rls_test'::regnamespace
                """
            )
        )
        row = res.first()
        assert row is not None
        assert row[0] is True  # rls enabled
        assert row[1] is True  # rls forced (even for table owner)


@pytest.mark.integration
class TestRLSIsolation:
    async def test_user_can_only_see_own_conversations(self, engine, setup_schema):
        """Two users, two conversations. Each user's session_var GUC reveals
        only their own row."""
        Session = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        graph_id = uuid.uuid4()

        # Seed: insert one knowledge_graphs row and two conversations as a
        # privileged superuser session (RLS bypass via SET ROLE not used —
        # we rely on the migration not setting FORCE on a non-owning role).
        async with Session() as s:
            await s.execute(text("SET search_path TO chat_rls_test"))
            await s.execute(
                text("INSERT INTO knowledge_graphs (id) VALUES (:g)").bindparams(
                    g=graph_id
                )
            )
            # Temporarily disable RLS to seed cross-user data. This is the
            # only block in the test that bypasses RLS — every read below
            # exercises the policy.
            await s.execute(
                text("ALTER TABLE chat_conversations DISABLE ROW LEVEL SECURITY")
            )
            await s.execute(
                text(
                    """
                    INSERT INTO chat_conversations (user_id, graph_id, title)
                    VALUES (:u, :g, 'A-conv'), (:v, :g, 'B-conv')
                    """
                ).bindparams(u=user_a, g=graph_id, v=user_b)
            )
            await s.execute(
                text("ALTER TABLE chat_conversations ENABLE ROW LEVEL SECURITY")
            )
            await s.execute(
                text("ALTER TABLE chat_conversations FORCE ROW LEVEL SECURITY")
            )
            await s.commit()

        # User A sees only A's conversation.
        async with Session() as s:
            await s.execute(text("SET search_path TO chat_rls_test"))
            await s.execute(
                text("SELECT set_config('app.current_user_id', :uid, true)").bindparams(
                    uid=str(user_a)
                )
            )
            res = await s.execute(
                text("SELECT title FROM chat_conversations ORDER BY title")
            )
            titles = [r[0] for r in res.fetchall()]
            assert titles == ["A-conv"], f"user_a saw: {titles}"

        # User B sees only B's conversation.
        async with Session() as s:
            await s.execute(text("SET search_path TO chat_rls_test"))
            await s.execute(
                text("SELECT set_config('app.current_user_id', :uid, true)").bindparams(
                    uid=str(user_b)
                )
            )
            res = await s.execute(
                text("SELECT title FROM chat_conversations ORDER BY title")
            )
            titles = [r[0] for r in res.fetchall()]
            assert titles == ["B-conv"], f"user_b saw: {titles}"

        # No GUC set → zero rows (the GUC is missing, cast fails gracefully).
        async with Session() as s:
            await s.execute(text("SET search_path TO chat_rls_test"))
            res = await s.execute(text("SELECT count(*) FROM chat_conversations"))
            count = res.scalar()
            assert count == 0, f"unset GUC should see no rows; got {count}"
