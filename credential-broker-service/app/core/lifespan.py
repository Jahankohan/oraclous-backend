from fastapi import FastAPI

from contextlib import asynccontextmanager
from app.core.config import settings
from app.repositories.credential_repository import CredentialRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the repository and store it in app state
    credential_repository = CredentialRepository(db_url=settings.DATABASE_URL)

    await credential_repository.create_tables()

    app.state.credential_repository = credential_repository
    print("Repository initialized")

    # Yield to allow the application to run
    yield

    # Shutdown: Close the repository (or any other resource cleanup)
    await app.state.credential_repository.close()
    print("Repository closed")
