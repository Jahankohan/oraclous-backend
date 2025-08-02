from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from app.models.user_model import User
from app.models.base_model import Base

class UserRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        await self.engine.dispose()

    async def get_user_by_id(self, user_id: str):
        async with self.Session() as session:
            result = await session.execute(select(User).where(User.id == user_id))
            return result.scalars().first()

    async def get_user_by_email(self, email: str):
        async with self.Session() as session:
            result = await session.execute(select(User).where(User.email == email))
            return result.scalars().first()

    async def create_user(self, email: str, first_name: str = None, last_name: str = None, profile_picture: str = None):
        async with self.Session() as session:
            user = User(email=email, first_name=first_name, last_name=last_name, profile_picture=profile_picture)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
    
    async def create_user_with_email(self, email: str, hashed_password: str):
        async with self.Session() as session:
            user = User(email=email, password_hash=hashed_password, is_email_verified=False, is_active=True, is_superuser=False)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
    
    async def set_new_verification_code(self, email: str):
        async with self.Session() as session:
            user = await session.execute(select(User).where(User.email == email))
            user = user.scalars().first()
            if user:
                user.set_verification_code()
                await session.commit()
                return user.verification_code
        return None

    async def verify_email(self, email: str, code: str) -> bool:
        async with self.Session() as session:
            user = await session.execute(select(User).where(User.email == email))
            user = user.scalars().first()
            if user and user.verification_code == code:
                user.is_email_verified = True
                user.verification_code = None
                await session.commit()
                return True
            return False

    async def update_password(self, email: str, new_hashed_password: str):
        async with self.Session() as session:
            user = await session.execute(select(User).where(User.email == email))
            user = user.scalars().first()
            if user:
                user.password_hash = new_hashed_password
                await session.commit()
