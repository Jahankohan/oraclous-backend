from typing import Dict, Any, Optional
from app.tools.registry import tool_registry
from app.interfaces.tool_executor import BaseToolExecutor
from app.schemas.tool_instance import ToolInstance, ExecutionContext


class ToolFactory:
    """
    Factory for creating and managing tool instances
    """
    
    @staticmethod
    def create_executor(tool_definition_id: str) -> BaseToolExecutor:
        """Create a tool executor from definition ID"""
        return tool_registry.create_executor(tool_definition_id)
    
    @staticmethod
    def validate_tool_instance(instance: ToolInstance) -> bool:
        """Validate that a tool instance is properly configured"""
        try:
            executor = ToolFactory.create_executor(instance.tool_definition_id)
            return True  # If we can create executor, basic validation passes
        except Exception:
            return False
    
    @staticmethod 
    async def execute_tool(
        instance: ToolInstance,
        input_data: Any,
        context: ExecutionContext
    ):
        """Execute a tool instance with given input and context"""
        executor = ToolFactory.create_executor(instance.tool_definition_id)

        print("Executor:", executor)  # Debugging line to check executor creation
        
        # Use async context manager if supported
        if hasattr(executor, '__aenter__'):
            async with executor as exec_instance:
                return await exec_instance.execute(input_data, context)
        else:
            return await executor.execute(input_data, context)

