# app/utils/tool_id_generator.py
import uuid
import hashlib
from typing import Optional


def generate_tool_id(
    name: str, 
    version: str = "1.0.0",
    category: str = "",
    namespace: Optional[str] = None
) -> uuid.UUID:  # FIXED: Return UUID object instead of string
    """
    Generate a deterministic UUID for a tool based on its characteristics.
    
    This ensures the same tool always gets the same ID across deployments,
    while maintaining UUID format compatibility with the database.
    
    Args:
        name: Tool name (e.g., "Google Drive Reader")
        version: Tool version (e.g., "1.0.0")
        category: Tool category (e.g., "INGESTION")
        namespace: Optional namespace for organization-specific tools
    
    Returns:
        UUID object that's deterministic based on input parameters
    """
    # Create a consistent string representation
    if namespace:
        tool_string = f"{namespace}:{name}:{version}:{category}"
    else:
        tool_string = f"oraclous:{name}:{version}:{category}"
    
    # Normalize the string (lowercase, strip whitespace)
    tool_string = tool_string.lower().strip().replace(" ", "-")
    
    # Generate deterministic UUID using namespace UUID5
    # Using a fixed namespace UUID for Oraclous tools
    ORACLOUS_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    
    tool_uuid = uuid.uuid5(ORACLOUS_NAMESPACE, tool_string)
    
    return tool_uuid  # Return UUID object directly


def generate_tool_id_from_class(tool_class) -> str:
    """
    Generate tool ID directly from a tool class.
    Extracts name and other info from the class's get_tool_definition method.
    
    Args:
        tool_class: Tool class with get_tool_definition class method
    
    Returns:
        String UUID for the tool
    """
    if not hasattr(tool_class, 'get_tool_definition'):
        raise ValueError(f"Tool class {tool_class.__name__} must have get_tool_definition method")
    
    # Get basic definition (without calling the full method to avoid circular dependency)
    class_name = tool_class.__name__
    
    # Create a simple mapping for common tools
    # This could be enhanced to extract from actual definition
    tool_info = {
        "GoogleDriveReader": ("Google Drive Reader", "INGESTION"),
        "NotionReader": ("Notion Reader", "INGESTION"), 
        "PostgreSQLReader": ("PostgreSQL Reader", "INGESTION"),
        "MySQLReader": ("MySQL Reader", "INGESTION"),
    }
    
    if class_name in tool_info:
        name, category = tool_info[class_name]
        return generate_tool_id(name, "1.0.0", category)
    else:
        # Fallback: use class name
        return generate_tool_id(class_name, "1.0.0", "UNKNOWN")


# Example usage and tool ID constants
class ToolIDs:
    """
    Centralized tool ID constants for easy reference
    Generated deterministically from tool characteristics
    """
    
    GOOGLE_DRIVE_READER = generate_tool_id("Google Drive Reader", "1.0.0", "INGESTION")
    NOTION_READER = generate_tool_id("Notion Reader", "1.0.0", "INGESTION")
    POSTGRESQL_READER = generate_tool_id("PostgreSQL Reader", "1.0.0", "INGESTION")
    MYSQL_READER = generate_tool_id("MySQL Reader", "1.0.0", "INGESTION")
    
    @classmethod
    def get_all_ids(cls) -> dict:
        """Get all tool IDs as a dictionary"""
        return {
            "google_drive_reader": cls.GOOGLE_DRIVE_READER,
            "notion_reader": cls.NOTION_READER,
            "postgresql_reader": cls.POSTGRESQL_READER,
            "mysql_reader": cls.MYSQL_READER,
        }


# Validation helper
def validate_tool_id_format(tool_id: str) -> bool:
    """
    Validate that a tool ID is a proper UUID format
    
    Args:
        tool_id: String to validate
        
    Returns:
        True if valid UUID format, False otherwise
    """
    try:
        uuid.UUID(tool_id)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # Test the ID generation
    print("🔧 Generated Tool IDs:")
    print(f"Google Drive Reader: {ToolIDs.GOOGLE_DRIVE_READER}")
    print(f"Notion Reader: {ToolIDs.NOTION_READER}")
    print(f"PostgreSQL Reader: {ToolIDs.POSTGRESQL_READER}")
    print(f"MySQL Reader: {ToolIDs.MYSQL_READER}")
    
    # Test consistency
    print("\n🔄 Testing consistency:")
    id1 = generate_tool_id("Google Drive Reader", "1.0.0", "INGESTION")
    id2 = generate_tool_id("Google Drive Reader", "1.0.0", "INGESTION")
    print(f"Same inputs produce same ID: {id1 == id2}")
    print(f"Generated ID: {id1}")
    
    # Test different versions
    print("\n📋 Testing version differences:")
    v1 = generate_tool_id("Google Drive Reader", "1.0.0", "INGESTION")
    v2 = generate_tool_id("Google Drive Reader", "2.0.0", "INGESTION")
    print(f"Different versions produce different IDs: {v1 != v2}")
    print(f"Version 1.0.0: {v1}")
    print(f"Version 2.0.0: {v2}")
