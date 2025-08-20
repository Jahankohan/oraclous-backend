from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import init_db, close_db
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    await init_db()

    # Patch tool registry from DB
    from app.core.database import get_session
    from app.tools.runtime_registry_patch import patch_registry_from_db
    async with get_session() as db:
        await patch_registry_from_db(db)

    yield
    # Shutdown
    print("Shutting down...")
    await close_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")
