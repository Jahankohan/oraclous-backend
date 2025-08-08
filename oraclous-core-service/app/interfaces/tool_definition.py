from abc import ABC, abstractmethod
from typing import Dict, Any
from app.schemas.tool_definition import ToolDefinition

# Abstract Base Classes
class BaseToolDefinition(ABC):
    """
    Abstract base class for tool definitions
    Handles tool metadata and validation
    """
    
    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Return the tool definition metadata"""
        pass
    
    @abstractmethod
    def validate_configuration(self, configuration: Dict[str, Any]) -> bool:
        """Validate tool configuration against schema"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data against input schema"""
        pass
    
    @abstractmethod
    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        """Validate provided credentials"""
        pass

