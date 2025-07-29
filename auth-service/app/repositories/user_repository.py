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
        # Close the connection to the database engine
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
