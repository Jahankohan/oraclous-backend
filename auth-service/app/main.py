
from fastapi import FastAPI
from app.routes import oauth
from app.core.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

@app.get("/", response_model=dict[str, str])
async def read_root() -> dict[str, str]:
    return {"status": "API service is running."}


app.include_router(oauth.router)
