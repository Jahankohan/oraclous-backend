from typing import Dict
from app.interfaces.tool_definition import ToolDefinition
from app.interfaces.tool_executor import BaseToolExecutor

# Helper classes for tool implementation
class ToolImplementationFactory:
    """Factory for creating tool executors from definitions"""
    
    _executors: Dict[str, type] = {}
    
    @classmethod
    def register_executor(cls, tool_type: str, executor_class: type):
        """Register an executor class for a tool type"""
        cls._executors[tool_type] = executor_class
    
    @classmethod
    def create_executor(cls, definition: ToolDefinition) -> BaseToolExecutor:
        """Create executor instance for a tool definition"""
        executor_class = cls._executors.get(definition.type)
        if not executor_class:
            raise ValueError(f"No executor registered for tool type: {definition.type}")
        return executor_class(definition)
