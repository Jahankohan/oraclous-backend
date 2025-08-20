# app/services/tool_sync_service.py
import logging
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from app.tools.registry import tool_registry
from app.services.tool_registry import ToolRegistryService
from app.schemas.tool_definition import ToolDefinition
from app.core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


class ToolSyncService:
    """
    Enhanced service to synchronize tool definitions between database and in-memory registry
    Now supports bidirectional sync: DB ↔ Memory
    """
    
    def __init__(self):
        self.synced_tools: List[str] = []
        self.failed_tools: List[Dict[str, Any]] = []
        self.skipped_tools: List[Dict[str, Any]] = []
        self.created_tools: List[str] = []  # NEW: Track DB-created tools
    
    async def sync_tools_on_startup(self) -> Dict[str, Any]:
        """
        Complete tool synchronization:
        1. Load memory tools from code
        2. Sync DB → Memory (existing functionality)
        3. Sync Memory → DB (NEW: auto-create missing DB tools)
        """
        logger.info("Starting complete tool synchronization...")
        
        try:
            async with AsyncSessionLocal() as session:
                db_registry = ToolRegistryService(session)
                
                # STEP 1: Get tools from both sources
                memory_tools = tool_registry.list_definitions()
                db_tools = await db_registry.list_tools(limit=1000)
                
                logger.info(f"Found {len(memory_tools)} tools in memory registry")
                logger.info(f"Found {len(db_tools)} tools in database")
                
                # STEP 2: Sync DB → Memory (existing functionality)
                for tool_definition in db_tools:
                    await self._sync_db_to_memory(tool_definition)
                
                # STEP 3: Sync Memory → DB (NEW: auto-create missing tools)
                await self._sync_memory_to_db(memory_tools, db_registry)
                
                # STEP 4: Build summary
                summary = {
                    "total_memory_tools": len(memory_tools),
                    "total_db_tools": len(db_tools),
                    "synced_successfully": len(self.synced_tools),  # FIXED: Use correct key
                    "created_in_db": len(self.created_tools),
                    "failed_tools": len(self.failed_tools),
                    "skipped_tools": len(self.skipped_tools),
                    "synced_tool_ids": self.synced_tools,
                    "created_tool_ids": self.created_tools,
                    "failed_tool_details": self.failed_tools,
                    "skipped_tool_details": self.skipped_tools
                }
                
                logger.info(
                    f"Tool synchronization completed: "
                    f"{summary['synced_successfully']} synced from DB, "
                    f"{summary['created_in_db']} created in DB, "
                    f"{summary['failed_tools']} failed, "
                    f"{summary['skipped_tools']} skipped"
                )
                
                return summary
                
        except Exception as e:
            logger.error(f"Tool synchronization failed: {str(e)}")
            raise
    
    async def _sync_memory_to_db(self, memory_tools: List[ToolDefinition], db_registry: ToolRegistryService):
        """
        NEW: Sync tools from memory to database
        Creates DB records for tools that exist in memory but not in DB
        """
        logger.info("Syncing memory tools to database...")
        
        for tool_definition in memory_tools:
            try:
                # Check if tool exists in DB
                existing_db_tool = await db_registry.get_tool(tool_definition.id)
                
                if existing_db_tool:
                    logger.debug(f"Tool {tool_definition.name} already exists in DB")
                    continue
                
                # Tool doesn't exist in DB - create it
                logger.info(f"Creating tool in DB: {tool_definition.name} ({tool_definition.id})")
                
                success = await db_registry.register_tool(tool_definition)
                
                if success:
                    self.created_tools.append(tool_definition.id)
                    logger.info(f"✅ Created tool in DB: {tool_definition.name}")
                else:
                    logger.error(f"❌ Failed to create tool in DB: {tool_definition.name}")
                    self.failed_tools.append({
                        "id": tool_definition.id,
                        "name": tool_definition.name,
                        "error": "Database registration failed",
                        "operation": "create_in_db"
                    })
                
            except Exception as e:
                logger.error(f"Error creating tool {tool_definition.name} in DB: {str(e)}")
                self.failed_tools.append({
                    "id": tool_definition.id,
                    "name": tool_definition.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "operation": "create_in_db"
                })
    
    async def _sync_db_to_memory(self, tool_definition: ToolDefinition):
        """
        Existing functionality: Sync single tool from DB to memory
        FIXED: Handle timezone-aware datetime comparisons
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
                "error_type": type(e).__name__,
                "operation": "sync_db_to_memory"
            })
            logger.error(f"Failed to sync tool {tool_definition.id}: {str(e)}")
    
    async def _handle_existing_tool(self, db_definition: ToolDefinition, memory_definition: ToolDefinition):
        """
        Handle case where tool exists in both DB and memory
        FIXED: Proper timezone handling for datetime comparison
        """
        try:
            # Normalize datetime objects to UTC for comparison
            db_updated = self._normalize_datetime(db_definition.updated_at)
            memory_updated = self._normalize_datetime(memory_definition.updated_at)
            
            if db_updated > memory_updated:
                # Update in-memory registry with DB version
                tool_registry._definitions[db_definition.id] = db_definition
                logger.debug(f"Updated tool {db_definition.id} in memory registry (DB version newer)")
            else:
                logger.debug(f"Kept existing tool {db_definition.id} in memory registry (memory version newer or same)")
                
        except Exception as e:
            logger.warning(f"Could not compare timestamps for tool {db_definition.id}: {str(e)}. Using DB version.")
            # Fallback: use DB version if comparison fails
            tool_registry._definitions[db_definition.id] = db_definition
    
    def _normalize_datetime(self, dt: datetime) -> datetime:
        """
        Normalize datetime to UTC timezone for safe comparison
        FIXED: Handle both timezone-aware and timezone-naive datetimes
        """
        if dt is None:
            return datetime.now(timezone.utc)
        
        if dt.tzinfo is None:
            # Timezone-naive datetime - assume UTC
            return dt.replace(tzinfo=timezone.utc)
        else:
            # Timezone-aware datetime - convert to UTC
            return dt.astimezone(timezone.utc)
    
    async def _register_db_tool(self, tool_definition: ToolDefinition, executor_class):
        """Register a tool from DB that doesn't exist in memory registry"""
        tool_registry._definitions[tool_definition.id] = tool_definition
        logger.info(f"Registered DB-only tool: {tool_definition.name}")

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