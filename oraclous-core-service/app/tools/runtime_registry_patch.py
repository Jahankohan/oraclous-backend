# app/tools/runtime_registry_patch.py
"""
Runtime patching for tool registry based on DB tool definitions.
"""
from app.tools.registry import tool_registry
from app.tools.implementations.ingestion.google_drive_reader import GoogleDriveReader
from app.tools.implementations.ingestion.mysql_reader import MySQLReader
from app.tools.implementations.ingestion.postgresql_reader import PostgreSQLReader
from app.tools.implementations.ingestion.notion_reader import NotionReader

# Mapping from DB tool definition IDs to Python classes
TOOL_CLASS_MAP = {
    "e64c5142-a080-4a29-ae01-6460600e0188": GoogleDriveReader,
    "455bb745-c101-43a5-b486-7658652b07c2": MySQLReader,
    "9e33aee0-aa97-4f7d-999f-8614349ee4ee": PostgreSQLReader,
    "de1bfc80-38f0-4819-b7b8-230b01f6ad49": NotionReader,
}

async def patch_registry_from_db(db):
    """
    Populate the runtime registry from DB tool definitions.
    """
    from app.repositories.tool_definition_repository import ToolDefinitionRepository
    repo = ToolDefinitionRepository(db)
    tool_definitions = await repo.get_all_definitions()
    print("Tool definitions:", tool_definitions)
    for tool_def in tool_definitions:
        tool_class = TOOL_CLASS_MAP.get(tool_def.id)
        print("Tool Class:", tool_class)
        if tool_class:
            tool_registry.register_tool(tool_class, definition=tool_def)
        else:
            print(f"Warning: No Python class found for tool definition ID {tool_def.id}")
