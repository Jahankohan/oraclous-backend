from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Async Redis client — shared singleton for the FastAPI process.
# Used by QueryCacheService and any other async callers.
# Celery workers should instantiate their own synchronous Redis client
# to avoid sharing an async connection across the fork boundary.
# ---------------------------------------------------------------------------
try:
    import redis.asyncio as _aioredis

    redis_client = _aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    logger.info("Async Redis client initialised (url=%s)", settings.REDIS_URL)
except Exception as _redis_init_err:  # pragma: no cover
    logger.warning(
        "Could not initialise async Redis client: %s — cache will be disabled",
        _redis_init_err,
    )
    redis_client = None  # type: ignore[assignment]


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models"""

    pass


# Database engine
engine = create_async_engine(
    settings.POSTGRES_URL,
    echo=settings.LOG_LEVEL == "DEBUG",
    pool_pre_ping=True,
    pool_recycle=300,
)

# Session factory
async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def _alembic_config():
    """Build an Alembic Config with absolute paths (cwd-independent)."""
    from pathlib import Path

    from alembic.config import Config

    repo_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(repo_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root / "alembic"))
    return cfg


async def _database_is_fresh() -> bool:
    """True when the DB has never been initialised (no alembic_version table)."""
    from sqlalchemy import text

    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'alembic_version')"
            )
        )
        return not bool(result.scalar())


async def init_database_schema() -> None:
    """Bring the SQL schema up to date. Alembic is the single source of truth.

    - Existing database: apply any pending migrations (``alembic upgrade head``).
    - Brand-new database: bootstrap the tables once from the ORM models, then
      stamp Alembic at ``head``.

    The app must NOT run ``Base.metadata.create_all`` over a live database on
    every startup. Doing so races migrations: a model-created table makes the
    matching ``CREATE TABLE`` migration fail with "already exists", leaving the
    schema half-owned (table present, ``alembic_version`` not advanced).
    """
    import asyncio

    # Import every ORM model so Base.metadata is complete before create_all
    # bootstraps a fresh database. A model module the app never imports
    # otherwise (e.g. an orphaned one) is invisible to create_all — its table
    # silently goes missing on a fresh install.
    import app.models  # noqa: F401
    from alembic import command

    cfg = _alembic_config()
    if await _database_is_fresh():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await asyncio.to_thread(command.stamp, cfg, "head")
        logger.info("Fresh database bootstrapped from models; Alembic stamped at head")
    else:
        await asyncio.to_thread(command.upgrade, cfg, "head")
        logger.info("Database schema up to date (alembic upgrade head)")


async def check_db_health() -> dict[str, str | bool]:
    """Check database connection health"""
    try:
        async with async_session_maker() as session:
            from sqlalchemy import text

            result = await session.execute(text("SELECT 1"))
            result.fetchone()
            return {"status": "healthy", "connected": True}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "connected": False, "error": str(e)}
