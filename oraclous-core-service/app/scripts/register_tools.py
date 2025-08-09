"""
Script to register all available tools with the database
"""
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import get_session, init_db
from app.services.tool_registry import ToolRegistryService
from app.tools.registry import tool_registry

# Import all tool implementations to trigger registration
from app.tools.implementations.ingestion.google_drive_reader import GoogleDriveReader
from app.tools.implementations.ingestion.notion_reader import NotionReader
from app.tools.implementations.ingestion.postgresql_reader import PostgreSQLReader
from app.tools.implementations.ingestion.mysql_reader import MySQLReader


async def register_all_tools():
    """Register all available tools in the database"""
    print("Starting tool registration...")
    
    # Initialize database
    await init_db()
    
    # Get database session
    async with get_session() as db_session:
        registry_service = ToolRegistryService(db_session)
        
        # Register each tool from the local registry
        registered_count = 0
        failed_count = 0
        
        for definition in tool_registry.list_definitions():
            try:
                success = await registry_service.register_tool(definition)
                if success:
                    print(f"✓ Registered: {definition.name}")
                    registered_count += 1
                else:
                    print(f"✗ Failed to register: {definition.name}")
                    failed_count += 1
            except Exception as e:
                print(f"✗ Error registering {definition.name}: {str(e)}")
                failed_count += 1
        
        print(f"\nRegistration complete:")
        print(f"  Successfully registered: {registered_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {registered_count + failed_count}")


def register_tools():
    """Register tools with the in-memory registry"""
    print("Registering tools with in-memory registry...")
    
    # Register Google Drive Reader
    tool_registry.register_tool(
        GoogleDriveReader,
        GoogleDriveReader.get_tool_definition()
    )
    
    # Register Notion Reader
    tool_registry.register_tool(
        NotionReader,
        NotionReader.get_tool_definition()
    )
    
    # Register PostgreSQL Reader
    tool_registry.register_tool(
        PostgreSQLReader,
        PostgreSQLReader.get_tool_definition()
    )
    
    # Register MySQL Reader
    tool_registry.register_tool(
        MySQLReader,
        MySQLReader.get_tool_definition()
    )
    
    print(f"Registered {len(tool_registry.list_definitions())} tools in memory")


if __name__ == "__main__":
    # Register tools in memory first
    register_tools()
    
    # Then register in database
    asyncio.run(register_all_tools())
