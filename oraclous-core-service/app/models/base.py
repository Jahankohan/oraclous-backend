from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from datetime import datetime
import uuid


class Base(AsyncAttrs, DeclarativeBase):
    """
    Base class for all database models
    Provides common fields and functionality
    """
    pass


class TimestampMixin:
    """
    Mixin to add created_at and updated_at timestamp fields
    FIXED: Ensure all timestamps are timezone-aware
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),  # This ensures timezone awareness
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),  # This ensures timezone awareness
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class UUIDMixin:
    """
    Mixin to add UUID primary key - FIXED to use proper UUID type
    """
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),  # FIXED: Use proper UUID type instead of String(36)
        primary_key=True,
        default=uuid.uuid4,  # FIXED: Use uuid.uuid4 directly
        nullable=False
    )
