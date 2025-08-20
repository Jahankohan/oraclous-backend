from abc import abstractmethod
from typing import Any, Dict, Optional
import asyncio
import httpx
from datetime import datetime
from decimal import Decimal

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
            # Basic input validation (lightweight)
            if not self.validate_input(input_data):
                return ExecutionResult(
                    success=False,
                    error_message="Invalid input data",
                    error_type="INVALID_INPUT"
                )
            
            # Basic context validation (lightweight)
            if not await self.validate_context(context):
                return ExecutionResult(
                    success=False,
                    error_message="Invalid execution context or missing credentials",
                    error_type="INVALID_CONTEXT"
                )
            
            # Execute the actual tool logic
            result = await self._execute_internal(input_data, context)
            
            # Calculate credits consumed
            credits = self.calculate_credits(input_data, result)
            
            if isinstance(credits, (int, float)):
                credits = Decimal(str(credits))
        
            result.credits_consumed = credits
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=f"Tool execution failed: {str(e)}",
                error_type=type(e).__name__,
                metadata={"error_details": str(e)}
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
        """
        Lightweight context validation (just check basic structure)
        Heavy validation is done by ValidationService before execution
        """
        if not context:
            return False
        
        # Just verify required context fields exist
        required_fields = ['instance_id', 'workflow_id', 'user_id', 'job_id']
        for field in required_fields:
            if not hasattr(context, field) or getattr(context, field) is None:
                return False
        
        return True
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Lightweight input validation (just check basic structure)
        Heavy validation should be done by the ValidationService
        """
        if input_data is None:
            return False
        
        # Basic type check - most tools expect dict input
        if not isinstance(input_data, dict):
            return False
        
        return True
    
    def get_credentials(self, context: ExecutionContext, cred_type: str) -> Optional[Dict[str, Any]]:
        """Helper to extract specific credential type"""
        if not context.credentials:
            return None
        return context.credentials.get(cred_type)
