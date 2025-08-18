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
