from fastapi import FastAPI
from app.routes import credential_routes
from app.core.lifespan import lifespan
from app.core.auth_middleware import AuthMiddleware

app = FastAPI(lifespan=lifespan)
app.add_middleware(AuthMiddleware)

app.include_router(credential_routes.router, prefix="/credentials", tags=["credentials"])
