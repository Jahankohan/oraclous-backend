import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI

from app.core.config import settings
from app.repositories.service_account_repository import ServiceAccountRepository
from app.repositories.token_repository import TokenRepository
from app.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the repository and store it in app state
    token_repository = TokenRepository(db_url=settings.DB_URL)
    user_repository = UserRepository(db_url=settings.DB_URL)
    sa_repository = ServiceAccountRepository(db_url=settings.DB_URL)

    await token_repository.create_tables()
    await user_repository.create_tables()
    await sa_repository.create_tables()

    app.state.token_repository = token_repository
    app.state.user_repository = user_repository
    app.state.sa_repository = sa_repository

    # Redis for rate limiting — non-fatal if unavailable at startup
    try:
        redis_client = aioredis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=False
        )
        await redis_client.ping()
        app.state.redis = redis_client
        logger.info("Redis connected for rate limiting: %s", settings.REDIS_URL)
    except Exception as exc:
        logger.warning("Redis unavailable at startup, rate limiting disabled: %s", exc)
        app.state.redis = None

    logger.info("Repository initialized")

    yield

    # Shutdown: Close the repository (or any other resource cleanup)
    await app.state.token_repository.close()
    await app.state.user_repository.close()
    await app.state.sa_repository.close()

    if app.state.redis is not None:
        await app.state.redis.aclose()

    logger.info("Repository closed")