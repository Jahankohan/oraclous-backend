from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app.core.config import settings
from app.core.lifespan import lifespan
from app.core.rate_limiter import limiter, rate_limit_exceeded_handler
from app.routes import auth_routes, oauth_routes

app = FastAPI(lifespan=lifespan)

# Register the slowapi limiter so decorators can find it
app.state.limiter = limiter
# Custom handler: never exposes rate-limit configuration in the response body
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Middleware order: last added = outermost (first to process requests).
# ProxyHeadersMiddleware must be outermost so it rewrites request.client
# before slowapi's key_func reads it for rate limiting.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    # allow_origin_regex covers production — oraclous.com and every company
    # tenant subdomain (company-name.oraclous.com), which a static list cannot
    # enumerate. allow_origins still covers explicit dev origins (localhost).
    allow_origin_regex=r"https://([a-z0-9-]+\.)?oraclous\.com",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
# Rewrite request.client to the real client IP from X-Forwarded-For,
# but only when the direct connection comes from a trusted proxy IP.
# This prevents X-Forwarded-For spoofing by untrusted callers.
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=settings.TRUSTED_PROXY_IPS)


@app.get("/", response_model=dict[str, str])
async def read_root() -> dict[str, str]:
    return {"status": "API service is running."}


app.include_router(oauth_routes.router)
app.include_router(auth_routes.router)
