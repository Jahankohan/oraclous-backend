from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from app.schemas.tool_instance import ToolInstance

class BaseInstanceManager(ABC):
    """
    Abstract base class for tool instance management
    """
    
    @abstractmethod
    async def create_instance(
        self, 
        tool_definition_id: str,
        workflow_id: str,
        configuration: Dict[str, Any]
    ) -> ToolInstance:
        """Create a new tool instance"""
        pass
    
    @abstractmethod
    async def configure_instance(
        self, 
        instance_id: str, 
        configuration: Dict[str, Any]
    ) -> bool:
        """Update instance configuration"""
        pass
    
    @abstractmethod
    async def get_instance(self, instance_id: str) -> Optional[ToolInstance]:
        """Retrieve tool instance by ID"""
        pass
    
    @abstractmethod
    async def validate_instance_ready(self, instance_id: str) -> bool:
        """Check if instance is ready for execution"""
        pass
