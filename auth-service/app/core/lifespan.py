from fastapi import FastAPI

from contextlib import asynccontextmanager
from app.core.config import settings
from app.repositories.token_repository import TokenRepository
from app.repositories.user_repository import UserRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the repository and store it in app state
    token_repository = TokenRepository(db_url=settings.DB_URL)
    user_repository = UserRepository(db_url=settings.DB_URL)

    await token_repository.create_tables()
    await user_repository.create_tables()

    app.state.token_repository = token_repository
    app.state.user_repository = user_repository
    print("Repository initialized")

    # Yield to allow the application to run
    yield

    # Shutdown: Close the repository (or any other resource cleanup)
    await app.state.token_repository.close()
    await app.state.user_repository.close()
    print("Repository closed")