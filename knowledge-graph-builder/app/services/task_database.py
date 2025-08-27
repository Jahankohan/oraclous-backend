"""
Database Session Management for Background Tasks
Provides clean database sessions with proper resource management
"""

from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class TaskDatabaseManager:
    """Manages database connections for background tasks"""
    
    @staticmethod
    @asynccontextmanager
    async def get_async_session():
        """
        Create an isolated async database session for background tasks
        
        Usage:
            async with TaskDatabaseManager.get_async_session() as session:
                # Use session for database operations
                result = await session.execute(query)
                await session.commit()
        """
        engine = None
        session = None
        
        try:
            # Create async engine for this task
            engine = create_async_engine(
                settings.POSTGRES_URL,
                pool_size=2,
                max_overflow=5,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            AsyncSessionLocal = sessionmaker(
                engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            # Create session
            session = AsyncSessionLocal()
            
            logger.debug("Created async database session for background task")
            
            yield session
            
        except Exception as e:
            if session:
                await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            if session:
                try:
                    await session.close()
                    logger.debug("Closed async database session")
                except Exception as e:
                    logger.warning(f"Error closing session: {e}")
            
            if engine:
                try:
                    await engine.dispose()
                    logger.debug("Disposed database engine")
                except Exception as e:
                    logger.warning(f"Error disposing engine: {e}")
    
    @staticmethod
    @asynccontextmanager
    async def get_sync_session():
        """
        Create a synchronous database session for background tasks
        
        Usage:
            async with TaskDatabaseManager.get_sync_session() as session:
                # Use session for database operations
                result = session.execute(query)
                session.commit()
        """
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        engine = None
        session = None
        
        try:
            # Convert async URL to sync for synchronous operations
            sync_url = settings.POSTGRES_URL.replace("+asyncpg", "")
            
            # Create sync engine
            engine = create_engine(
                sync_url,
                pool_size=3,
                max_overflow=5,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Create session factory
            SyncSessionLocal = sessionmaker(bind=engine)
            
            # Create session
            session = SyncSessionLocal()
            
            logger.debug("Created sync database session for background task")
            
            yield session
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            if session:
                try:
                    session.close()
                    logger.debug("Closed sync database session")
                except Exception as e:
                    logger.warning(f"Error closing session: {e}")
            
            if engine:
                try:
                    engine.dispose()
                    logger.debug("Disposed sync database engine")
                except Exception as e:
                    logger.warning(f"Error disposing engine: {e}")
