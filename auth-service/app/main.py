
from fastapi import FastAPI
from app.routes import oauth_routes
from app.routes import auth_routes 
from app.core.lifespan import lifespan
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)
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