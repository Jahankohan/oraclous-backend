from abc import ABC, abstractmethod
from typing import Any
from app.schemas.tool_definition import ToolDefinition
from app.schemas.tool_instance import ExecutionContext, ExecutionResult, ToolInstance

class BaseToolExecutor(ABC):
    """
    Abstract base class for tool execution
    Handles the actual tool logic
    """
    
    def __init__(self, definition: ToolDefinition):
        self.definition = definition
    
    @abstractmethod
    async def execute(
        self, 
        input_data: Any, 
        context: ExecutionContext
    ) -> ExecutionResult:
        """Execute the tool with given input and context"""
        pass
    
    @abstractmethod
    async def validate_instance(self, instance: ToolInstance) -> bool:
        """Validate tool instance configuration"""
        pass
    
    def calculate_credits(self, input_data: Any, result: ExecutionResult) -> float:
        """Calculate credits consumed (can be overridden)"""
        return 1.0  # Default: 1 credit per execution
