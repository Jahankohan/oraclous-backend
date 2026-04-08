
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from app.core.lifespan import lifespan
from app.core.rate_limiter import limiter, rate_limit_exceeded_handler
from app.routes import auth_routes, oauth_routes

app = FastAPI(lifespan=lifespan)

# Register the slowapi limiter so decorators can find it
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict[str, str])
async def read_root() -> dict[str, str]:
    return {"status": "API service is running."}


app.include_router(oauth_routes.router)
app.include_router(auth_routes.router)