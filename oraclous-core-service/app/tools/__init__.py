"""
Tools module initialization
Automatically registers all available tools
"""
from app.tools.registry import ToolRegistry
from app.tools.factory import ToolFactory

# Import and register all tools
def _register_all_tools():
    """Register all available tools"""
    try:
        from app.tools.implementations.ingestion.google_drive_reader import GoogleDriveReader
        ToolRegistry.register_tool(GoogleDriveReader, GoogleDriveReader.get_tool_definition())
    except ImportError as e:
        print(f"Warning: Could not register GoogleDriveReader: {e}")
    
    try:
        from app.tools.implementations.ingestion.notion_reader import NotionReader
        ToolRegistry.register_tool(NotionReader, NotionReader.get_tool_definition())
    except ImportError as e:
        print(f"Warning: Could not register NotionReader: {e}")
    
    try:
        from app.tools.implementations.ingestion.postgresql_reader import PostgreSQLReader
        ToolRegistry.register_tool(PostgreSQLReader, PostgreSQLReader.get_tool_definition())
    except ImportError as e:
        print(f"Warning: Could not register PostgreSQLReader: {e}")
    
    try:
        from app.tools.implementations.ingestion.mysql_reader import MySQLReader
        ToolRegistry.register_tool(MySQLReader, MySQLReader.get_tool_definition())
    except ImportError as e:
        print(f"Warning: Could not register MySQLReader: {e}")

# Auto-register tools on module import
# _register_all_tools()

# Export main classes
# __all__ = ["ToolRegistry", "ToolFactory"]
