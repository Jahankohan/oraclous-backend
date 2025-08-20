# app/core/lifespan.py
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.services.tool_sync_service import tool_sync_service
from app.core.database import init_db, close_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for startup and shutdown events
    """
    # Startup
    logger.info("Starting Oraclous Core Service...")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        await init_db()
        
        # Sync tools from database to in-memory registry
        logger.info("Synchronizing tools...")
        sync_result = await tool_sync_service.sync_tools_on_startup()
        
        # Store sync result in app state for monitoring
        app.state.tool_sync_result = sync_result
        
        logger.info(
            f"Startup completed successfully. Synced {sync_result['synced_successfully']} tools."
        )
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        # You might want to raise this to prevent app startup
        # raise
        
    yield
    
    # Shutdown
    logger.info("Shutting down Oraclous Core Service...")
    
    try:
        # Close database connections
        await close_db()
        logger.info("Database connections closed.")
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")
    
    logger.info("Shutdown completed.")


# Alternative approach using traditional startup/shutdown events
# (Use this if you prefer not to use lifespan context manager)

async def startup_event():
    """Startup event handler"""
    logger.info("Starting Oraclous Core Service...")
    
    try:
        await init_db()
        sync_result = await tool_sync_service.sync_tools_on_startup()
        logger.info(f"Startup completed. Synced {sync_result['synced_successfully']} tools.")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")


async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down Oraclous Core Service...")
    
    try:
        await close_db()
        logger.info("Shutdown completed.")
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")


# Update your main.py or app creation file:
"""
# In your main.py or wherever you create the FastAPI app:

from fastapi import FastAPI
from app.core.lifespan import lifespan

app = FastAPI(
    title="Oraclous Core Service",
    version="1.0.0",
    lifespan=lifespan
)

# OR if using traditional events:
# app.add_event_handler("startup", startup_event)  
# app.add_event_handler("shutdown", shutdown_event)
"""