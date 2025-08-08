from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import DateTime, String
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional


class Base(AsyncAttrs, DeclarativeBase):
    """
    Base class for all database models
    Provides common fields and functionality
    """
    pass


class TimestampMixin:
    """
    Mixin to add created_at and updated_at timestamp fields
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class UUIDMixin:
    """
    Mixin to add UUID primary key
    """
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True,
        nullable=False
    )
