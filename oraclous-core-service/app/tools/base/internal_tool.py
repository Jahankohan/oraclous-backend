from abc import abstractmethod
from typing import Any, Dict, Optional
import asyncio
import httpx
from datetime import datetime

from app.interfaces.tool_executor import BaseToolExecutor
from app.schemas.tool_instance import ExecutionContext, ExecutionResult
from app.schemas.tool_definition import ToolDefinition
from app.utils.validation import ToolValidationMixin


class InternalTool(BaseToolExecutor, ToolValidationMixin):
    """
    Base class for internal tool implementations
    These are tools implemented directly in our codebase
    """
    
    def __init__(self, definition: ToolDefinition):
        super().__init__(definition)
        self.http_client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.http_client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.http_client:
            await self.http_client.aclose()
    
    async def execute(
        self, 
        input_data: Any, 
        context: ExecutionContext
    ) -> ExecutionResult:
        """Main execution method with error handling"""
        try:
            # Validate inputs
            if not self.validate_input(input_data):
                return ExecutionResult(
                    success=False,
                    error_message="Invalid input data"
                )
            
            # Validate context
            if not await self.validate_context(context):
                return ExecutionResult(
                    success=False,
                    error_message="Invalid execution context or missing credentials"
                )
            
            # Execute the actual tool logic
            result = await self._execute_internal(input_data, context)
            
            # Calculate credits consumed
            credits = self.calculate_credits(input_data, result)
            result.credits_consumed = credits
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=f"Tool execution failed: {str(e)}",
                metadata={"error_type": type(e).__name__}
            )
    
    @abstractmethod
    async def _execute_internal(
        self, 
        input_data: Any, 
        context: ExecutionContext
    ) -> ExecutionResult:
        """Internal execution logic - to be implemented by subclasses"""
        pass
    
    async def validate_context(self, context: ExecutionContext) -> bool:
        """Validate execution context and credentials"""
        # Check if required credentials are present
        for req in self.definition.credential_requirements:
            if req.required and not context.credentials:
                return False
            
            if req.required and req.type.value not in context.credentials:
                return False
        
        return True
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data against schema"""
        # Use the mixin method to validate against input schema
        return self.validate_schema_match(input_data, self.definition.input_schema)
    
    def get_credentials(self, context: ExecutionContext, cred_type: str) -> Optional[Dict[str, Any]]:
        """Helper to extract specific credential type"""
        if not context.credentials:
            return None
        return context.credentials.get(cred_type)
