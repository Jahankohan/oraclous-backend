from app.models.base_model import Base
from sqlalchemy import Column, String, Boolean, TIMESTAMP, func
from datetime import datetime, timezone, timedelta
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=True)
    is_email_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    verification_code = Column(String, nullable=True)
    verification_code_expiry = Column(TIMESTAMP(timezone=True), nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    profile_picture = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def set_verification_code(self):
        from random import randint
        code = randint(100000, 999999)
        self.verification_code = str(code)
        self.verification_code_expiry = datetime.now(timezone.utc) + timedelta(minutes=60)
