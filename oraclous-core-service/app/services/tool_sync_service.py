# app/services/tool_sync_service.py
import logging
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.registry import tool_registry
from app.services.tool_registry import ToolRegistryService
from app.schemas.tool_definition import ToolDefinition
from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


class ToolSyncService:
    """
    Service to synchronize tool definitions between database and in-memory registry
    Used during application startup to ensure all DB tools are available in memory
    """
    
    def __init__(self):
        self.synced_tools: List[str] = []
        self.failed_tools: List[Dict[str, Any]] = []
        self.skipped_tools: List[Dict[str, Any]] = []
    
    async def sync_tools_on_startup(self) -> Dict[str, Any]:
        """
        Main method to sync all tools from DB to in-memory registry
        Returns summary of sync operation
        """
        logger.info("Starting tool synchronization from database to in-memory registry...")
        
        try:
            # Create database session
            async with AsyncSessionLocal() as session:
                db_registry = ToolRegistryService(session)
                
                # Get all tool definitions from database
                db_tools = await db_registry.list_tools(limit=1000)  # Get all tools
                
                logger.info(f"Found {len(db_tools)} tool definitions in database")
                
                # Process each tool
                for tool_definition in db_tools:
                    await self._sync_single_tool(tool_definition)
                
                # Log summary
                summary = {
                    "total_db_tools": len(db_tools),
                    "synced_successfully": len(self.synced_tools),
                    "failed_tools": len(self.failed_tools),
                    "skipped_tools": len(self.skipped_tools),
                    "synced_tool_ids": self.synced_tools,
                    "failed_tool_details": self.failed_tools,
                    "skipped_tool_details": self.skipped_tools
                }
                
                logger.info(
                    f"Tool synchronization completed: {summary['synced_successfully']}/{summary['total_db_tools']} "
                    f"tools synced successfully"
                )
                
                if self.failed_tools:
                    logger.warning(f"Failed to sync {len(self.failed_tools)} tools")
                
                if self.skipped_tools:
                    logger.info(f"Skipped {len(self.skipped_tools)} tools (no implementation)")
                
                return summary
                
        except Exception as e:
            logger.error(f"Tool synchronization failed: {str(e)}")
            raise
    
    async def _sync_single_tool(self, tool_definition: ToolDefinition):
        """
        Sync a single tool definition to in-memory registry
        """
        try:
            tool_id = tool_definition.id
            
            # Check if implementation exists in in-memory registry
            executor_class = tool_registry.get_executor_class(tool_id)
            
            if executor_class is None:
                # No implementation found - this is expected for some tools
                self.skipped_tools.append({
                    "id": tool_id,
                    "name": tool_definition.name,
                    "reason": "No implementation class found in in-memory registry",
                    "category": tool_definition.category.value,
                    "type": tool_definition.type.value
                })
                logger.debug(f"Skipped tool {tool_id} - no implementation available")
                return
            
            # Check if definition already exists in memory (from code registration)
            existing_definition = tool_registry.get_definition(tool_id)
            
            if existing_definition:
                # Compare versions or update if DB is newer
                await self._handle_existing_tool(tool_definition, existing_definition)
            else:
                # Register new tool from DB
                await self._register_db_tool(tool_definition, executor_class)
            
            self.synced_tools.append(tool_id)
            logger.debug(f"Successfully synced tool: {tool_definition.name} ({tool_id})")
            
        except Exception as e:
            self.failed_tools.append({
                "id": tool_definition.id,
                "name": tool_definition.name,
                "error": str(e),
                "error_type": type(e).__name__
            })
            logger.error(f"Failed to sync tool {tool_definition.id}: {str(e)}")
    
    async def _handle_existing_tool(
        self, 
        db_definition: ToolDefinition, 
        memory_definition: ToolDefinition
    ):
        """
        Handle case where tool exists in both DB and memory
        Decide whether to update or keep existing
        """
        # For now, prefer DB version (assuming it's more up-to-date)
        # In production, you might want more sophisticated version comparison
        
        db_updated = db_definition.updated_at
        memory_updated = memory_definition.updated_at
        
        if db_updated > memory_updated:
            # Update in-memory registry with DB version
            tool_registry._definitions[db_definition.id] = db_definition
            logger.debug(
                f"Updated tool {db_definition.id} in memory registry "
                f"(DB version newer: {db_updated} > {memory_updated})"
            )
        else:
            logger.debug(
                f"Kept existing tool {db_definition.id} in memory registry "
                f"(memory version newer or same)"
            )
    
    async def _register_db_tool(self, tool_definition: ToolDefinition, executor_class):
        """
        Register a tool from DB that doesn't exist in memory registry
        """
        # This should not happen in normal flow since implementations 
        # should register themselves, but we handle it for completeness
        
        tool_registry._definitions[tool_definition.id] = tool_definition
        # Note: executor_class should already be registered if we found it
        
        logger.info(f"Registered DB-only tool: {tool_definition.name}")
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """
        Get current synchronization status
        """
        memory_tools = tool_registry.list_definitions()
        
        return {
            "last_sync_stats": {
                "synced_tools": len(self.synced_tools),
                "failed_tools": len(self.failed_tools),
                "skipped_tools": len(self.skipped_tools)
            },
            "current_memory_tools": len(memory_tools),
            "memory_tool_ids": [tool.id for tool in memory_tools],
            "failed_tools": self.failed_tools,
            "skipped_tools": self.skipped_tools
        }
    
    async def validate_tool_availability(self, tool_id: str) -> Dict[str, Any]:
        """
        Validate if a tool is available for execution
        """
        # Check in-memory registry
        definition = tool_registry.get_definition(tool_id)
        executor_class = tool_registry.get_executor_class(tool_id)
        
        # Check database
        async with AsyncSessionLocal() as session:
            db_registry = ToolRegistryService(session)
            db_definition = await db_registry.get_tool(tool_id)
        
        return {
            "tool_id": tool_id,
            "definition_in_memory": definition is not None,
            "executor_in_memory": executor_class is not None,
            "definition_in_db": db_definition is not None,
            "available_for_execution": definition is not None and executor_class is not None,
            "definition_source": self._get_definition_source(definition, db_definition),
            "implementation_available": executor_class is not None,
            "implementation_class": executor_class.__name__ if executor_class else None
        }
    
    def _get_definition_source(self, memory_def, db_def) -> str:
        """Determine where the current definition comes from"""
        if memory_def and db_def:
            return "both (memory used)"
        elif memory_def:
            return "memory_only"
        elif db_def:
            return "db_only"
        else:
            return "not_found"


# Global instance
tool_sync_service = ToolSyncService()