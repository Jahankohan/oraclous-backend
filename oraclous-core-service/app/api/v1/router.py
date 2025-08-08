from fastapi import APIRouter
from app.api.v1.endpoints import tools

router = APIRouter()

router.include_router(tools.router)