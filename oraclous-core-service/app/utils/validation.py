from typing import Any, Dict, List
from app.schemas.tool_definition import ToolSchema

# Example usage interfaces
class ToolValidationMixin:
    """Mixin providing common validation utilities"""
    
    def validate_schema_match(self, data: Any, schema: ToolSchema) -> bool:
        """Validate data against schema"""
        # Implementation would use jsonschema or similar
        pass
    
    def validate_required_fields(self, data: Dict, required: List[str]) -> bool:
        """Validate required fields are present"""
        return all(field in data for field in required)
