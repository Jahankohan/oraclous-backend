# app/utils/validation.py - CREATE THIS FILE
from typing import Any, Dict
import jsonschema


class ToolValidationMixin:
    """
    Mixin to provide validation capabilities for tools
    """
    
    def validate_schema_match(self, data: Any, schema: Dict[str, Any]) -> bool:
        """
        Validate data against a JSON schema
        """
        try:
            # Convert ToolSchema to dict if needed
            if hasattr(schema, 'dict'):
                schema_dict = schema.dict()
            else:
                schema_dict = schema
            
            # Basic validation - just check if it's a dict for now
            # In production, you'd use proper JSON schema validation
            if schema_dict.get("type") == "object" and not isinstance(data, dict):
                return False
            
            # Check required fields if specified
            required_fields = schema_dict.get("required", [])
            if required_fields and isinstance(data, dict):
                for field in required_fields:
                    if field not in data:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def validate_tool_definition(self, definition) -> bool:
        """Validate tool definition structure"""
        if not definition:
            return False
        
        # Basic checks
        if not hasattr(definition, 'name') or not definition.name:
            return False
        
        if not hasattr(definition, 'input_schema') or not definition.input_schema:
            return False
        
        if not hasattr(definition, 'output_schema') or not definition.output_schema:
            return False
        
        return True
