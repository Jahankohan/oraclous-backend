from typing import Dict, Any, Optional
from app.tools.base.internal_tool import InternalTool
from app.schemas.tool_instance import ExecutionContext


class DatabaseTool(InternalTool):
    """
    Base class for database connection tools
    """
    
    async def validate_context(self, context: ExecutionContext) -> bool:
        """Validate database credentials"""
        if not await super().validate_context(context):
            return False
        
        db_creds = self.get_credentials(context, "CONNECTION_STRING")
        if not db_creds:
            return False
        
        connection_string = db_creds.get("connection_string")
        if not connection_string:
            return False
        
        return True
    
    def get_connection_string(self, context: ExecutionContext) -> str:
        """Get database connection string"""
        db_creds = self.get_credentials(context, "CONNECTION_STRING")
        if not db_creds:
            raise ValueError("Database credentials not found")
        
        return db_creds["connection_string"]
    
    async def test_connection(self, context: ExecutionContext) -> bool:
        """Test database connection"""
        try:
            connection_string = self.get_connection_string(context)
            # This would be implemented based on the database type
            # For now, just return True if connection string exists
            return bool(connection_string)
        except Exception:
            return False
