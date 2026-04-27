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


async def create_tables():
    """Create all tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")


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
