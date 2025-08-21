from neo4j import AsyncGraphDatabase, AsyncDriver
from typing import Optional, Dict, Any, List
import asyncio
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class Neo4jClient:
    """Neo4j database client with async support"""
    
    def __init__(self):
        self.driver: Optional[AsyncDriver] = None
        self._lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Establish connection to Neo4j"""
        if self.driver is None:
            async with self._lock:
                if self.driver is None:
                    try:
                        self.driver = AsyncGraphDatabase.driver(
                            settings.NEO4J_URI,
                            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
                        )
                        # Test connection
                        await self.driver.verify_connectivity()
                        logger.info("Successfully connected to Neo4j")
                    except Exception as e:
                        logger.error(f"Failed to connect to Neo4j: {e}")
                        raise
    
    async def disconnect(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")
    
    async def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        if not self.driver:
            await self.connect()
        
        db_name = database or settings.NEO4J_DATABASE
        
        try:
            async with self.driver.session(database=db_name) as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                return records
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    async def execute_write_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a write query in a transaction"""
        if not self.driver:
            await self.connect()
        
        db_name = database or settings.NEO4J_DATABASE
        
        try:
            async with self.driver.session(database=db_name) as session:
                result = await session.execute_write(
                    self._execute_query_tx, query, parameters or {}
                )
                return result
        except Exception as e:
            logger.error(f"Write query execution failed: {e}")
            raise
    
    @staticmethod
    async def _execute_query_tx(tx, query: str, parameters: Dict[str, Any]):
        """Transaction function for write queries"""
        result = await tx.run(query, parameters)
        return await result.data()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j connection health"""
        try:
            if not self.driver:
                await self.connect()
            
            result = await self.execute_query("RETURN 1 as health")
            
            # Get database info
            db_info = await self.execute_query(
                "CALL dbms.components() YIELD name, versions, edition"
            )
            
            return {
                "status": "healthy",
                "connected": True,
                "database_info": db_info[0] if db_info else None,
                "uri": settings.NEO4J_URI
            }
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "uri": settings.NEO4J_URI
            }

# Global client instance
neo4j_client = Neo4jClient()
