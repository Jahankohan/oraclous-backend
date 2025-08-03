from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
import httpx
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        internal_token = request.headers.get("x-internal-service-key")
        auth_header = request.headers.get("Authorization")

        # Skip authentication for public endpoints if needed
        if request.url.path.startswith("/health") or request.url.path.startswith("/docs"):
            return await call_next(request)
        
        if internal_token:
            if internal_token != settings.INTERNAL_SERVICE_KEY:
                return JSONResponse({"detail": "Invalid internal token"}, status_code=401)
            return await call_next(request)

        logger.info(f"Received Authorization header: {auth_header}")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse({"detail": "Missing or invalid Authorization header"}, status_code=401)

        token = auth_header.split(" ")[1]

        # Delegate validation to Auth Service
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.AUTH_SERVICE_URL}/auth/me",
                headers={"Authorization": auth_header}  # Pass the full Authorization header
            )

        if response.status_code != 200:
            return JSONResponse({"detail": "Invalid or expired token"}, status_code=401)

        user_data = response.json()
        request.state.user = user_data  # Attach validated user details

        return await call_next(request)
