"""Pydantic schemas for the entity listing API (ORA-372)."""

from pydantic import BaseModel


class EntityItem(BaseModel):
    id: str
    name: str | None
    type: str | None
    confidence: float
    community_id: str | None
    degree: int


class EntityListResponse(BaseModel):
    items: list[EntityItem]
    total: int
    page: int
    page_size: int
