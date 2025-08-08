from fastapi import FastAPI

from app.routes import credential_routes
from app.core.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

app.include_router(credential_routes.router, prefix="/credentials", tags=["credentials"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "credential-broker"}