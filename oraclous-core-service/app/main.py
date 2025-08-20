from fastapi import FastAPI

from app.core.lifespan import lifespan
from app.core.config import settings
from app.api.v1.router import api_router




app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")
