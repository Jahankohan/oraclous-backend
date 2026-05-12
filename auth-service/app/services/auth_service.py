import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, status
from passlib.context import CryptContext

from app.core.config import settings
from app.core.jwt_handler import (
    create_access_token,
    create_refresh_token,
    verify_access_token,
    verify_refresh_token,
)
from app.models.user_model import User
from app.repositories.user_repository import UserRepository
from app.schema import auth_schemas
from app.services.email_service import send_email

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
logging.basicConfig(level=logging.DEBUG)


class AuthService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    async def get_user_by_email(self, email: str):
        user = await self.repository.get_user_by_email(email)
        return user

    async def create_user(
        self, user: auth_schemas.UserCreateWithEmail
    ) -> auth_schemas.Token:
        db_user = await self.get_user_by_email(user.email)
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
            )
        hashed_password = pwd_context.hash(user.password)
        db_user = await self.repository.create_user_with_email(
            user.email, hashed_password
        )

        verification_code = await self.repository.set_new_verification_code(
            db_user.email
        )

        body = f"Your verification code is: {verification_code}"
        await send_email(db_user.email, "Oraclous Verification Code", body)

        access_token, expires_in = await self.create_access_token(
            data={
                "sub": str(db_user.id),
                "email": db_user.email,
                "is_superuser": db_user.is_superuser,
            }
        )
        refresh_token = await self.create_refresh_token(
            data={
                "sub": str(db_user.id),
                "email": db_user.email,
                "is_superuser": db_user.is_superuser,
            }
        )
        token_schema = auth_schemas.Token(
            email=db_user.email,
            is_superuser=db_user.is_superuser,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            token_type="bearer",
        )
        return token_schema

    async def authenticate_user(self, email: str, password: str):
        db_user = await self.get_user_by_email(email)
        if not db_user or not pwd_context.verify(password, db_user.password_hash):
            return None

        access_token, expires_in = await self.create_access_token(
            data={
                "sub": str(db_user.id),
                "email": db_user.email,
                "is_superuser": db_user.is_superuser,
            }
        )
        refresh_token = await self.create_refresh_token(
            data={
                "sub": str(db_user.id),
                "email": db_user.email,
                "is_superuser": db_user.is_superuser,
            }
        )
        token_schema = auth_schemas.Token(
            email=db_user.email,
            is_superuser=db_user.is_superuser,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            token_type="bearer",
        )
        return token_schema

    async def send_password_reset_email(self, email: str):
        db_user = await self.get_user_by_email(email)
        if not db_user:
            raise None

        reset_token, _ = await self.create_access_token(
            data={
                "sub": str(db_user.id),
                "email": db_user.email,
                "is_superuser": db_user.is_superuser,
            },
            expires_delta=timedelta(minutes=15),
        )
        reset_link = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
        print("Reset link:", reset_link)  # Debugging line
        body = f"Click the following link to reset your password: {reset_link}"
        await send_email(db_user.email, "Oraclous Password Reset Link \n", body)
        return {"message": "Password reset email sent"}

    async def reset_password(self, token: str, new_password: str):
        payload = await self.verify_access_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

        db_user = await self.repository.get_user_by_id(user_id)
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        hashed_password = pwd_context.hash(new_password)
        await self.repository.update_password(db_user.email, hashed_password)
        return {"message": "Password reset successful"}

    async def change_password(self, user: User, new_password: str):
        hashed_password = pwd_context.hash(new_password)
        await self.repository.update_password(user.email, hashed_password)
        return {"message": "Password changed successfully"}

    async def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ):
        return create_access_token(data, expires_delta)

    async def resend_verification_email(self, token: str):
        print("Resending verification email with token:", token)
        payload = await self.verify_access_token(token)

        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

        db_user = await self.repository.get_user_by_id(user_id)
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        verification_code = await self.repository.set_new_verification_code(
            db_user.email
        )

        body = f"Your verification code is: {verification_code}"
        await send_email(db_user.email, "Oraclous Verification Code", body)
        return {"message": "Verification email resent successfully"}

    async def verify_user_email(self, token: str, code: int):
        payload = await self.verify_access_token(token)
        print("Payload from token:", payload)
        user_id = payload.get("sub")
        print("User ID from token:", user_id)

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

        db_user = await self.repository.get_user_by_id(user_id)
        if not db_user:
            raise HTTPException(status_code=400, detail="User not found")

        if db_user.verification_code != code or datetime.now(
            timezone.utc
        ) > db_user.verification_code_expiry.replace(tzinfo=timezone.utc):
            raise HTTPException(
                status_code=400, detail="Invalid or expired verification code"
            )

        await self.repository.verify_email(db_user.email, db_user.verification_code)
        return {"message": "Email verified successfully"}

    async def verify_access_token(self, token: str):
        return verify_access_token(token=token)

    async def create_refresh_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ):
        return create_refresh_token(data, expires_delta)

    async def verify_refresh_token(self, token: str):
        payload = verify_refresh_token(token)
        db_user = await self.repository.get_user_by_id(payload.user_id)

        access_token, expires_in = await self.create_access_token(
            data={
                "sub": str(db_user.id),
                "email": db_user.email,
                "is_superuser": db_user.is_superuser,
            }
        )
        refresh_token = await self.create_refresh_token(
            data={
                "sub": str(db_user.id),
                "email": db_user.email,
                "is_superuser": db_user.is_superuser,
            }
        )

        token_schema = auth_schemas.Token(
            email=db_user.email,
            is_superuser=db_user.is_superuser,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in,
            token_type="bearer",
        )
        return token_schema
