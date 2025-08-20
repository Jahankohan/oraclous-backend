# app/tools/registry.py
from typing import Dict, Type, List
from app.interfaces.tool_executor import BaseToolExecutor
from app.schemas.tool_definition import ToolDefinition


class ToolRegistry:
    """
    Registry for managing tool definitions and their implementations
    """
    
    def __init__(self):
        self._definitions: Dict[str, ToolDefinition] = {}
        self._executors: Dict[str, Type[BaseToolExecutor]] = {}
    
    def register_tool(
        self, 
        tool_class: Type[BaseToolExecutor],
        definition: ToolDefinition = None
    ):
        """Register a tool class with its definition"""
        if definition is None:
            # Try to get definition from class method
            if hasattr(tool_class, 'get_tool_definition'):
                definition = tool_class.get_tool_definition()
            else:
                raise ValueError(f"Tool class {tool_class.__name__} must provide definition")
        
        self._definitions[definition.id] = definition
        self._executors[definition.id] = tool_class
        
        print(f"Registered tool: {definition.name} ({definition.id})")
    
    def get_definition(self, tool_id: str) -> ToolDefinition:
        """Get tool definition by ID"""
        return self._definitions.get(tool_id)
    
    def get_executor_class(self, tool_id: str) -> Type[BaseToolExecutor]:
        """Get executor class by tool ID"""
        return self._executors.get(tool_id)
    
    def list_definitions(self) -> List[ToolDefinition]:
        """List all registered tool definitions"""
        return list(self._definitions.values())
    
    def create_executor(self, tool_id: str) -> BaseToolExecutor:
        """Create executor instance for a tool"""
        executor_class = self.get_executor_class(tool_id)
        if not executor_class:
            raise ValueError(f"No executor found for tool: {tool_id}")
        
        definition = self.get_definition(tool_id)
        print("Definition:", definition)
        return executor_class(definition)


# Global tool registry instance
tool_registry = ToolRegistry()