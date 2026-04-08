from neo4j import AsyncGraphDatabase, GraphDatabase, AsyncDriver, Driver
from typing import Optional, Dict, Any, List
import asyncio
from opentelemetry import trace as otel_trace
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_tracer = otel_trace.get_tracer("oraclous.neo4j")

class Neo4jClient:
    """
    Enhanced Neo4j database client with dual driver support.
    
    Provides both asynchronous (for FastAPI endpoints) and synchronous 
    (for neo4j_graphrag components) drivers to handle different usage patterns
    in distributed architecture with FastAPI and Celery workers.
    
    Features:
        - Async driver for FastAPI endpoints and async operations
        - Sync driver for neo4j_graphrag components (VectorRetriever, etc.)
        - Thread-safe connection management with locks
        - Automatic index creation for multi-tenant graph isolation
        - Health checking for both driver types
    
    Examples:
        >>> client = Neo4jClient()
        >>> await client.connect_async()  # For FastAPI
        >>> client.connect_sync()         # For GraphRAG
        >>> 
        >>> # Use async driver for FastAPI operations
        >>> async with client.async_driver.session() as session:
        ...     result = await session.run("RETURN 1")
        >>>
        >>> # Use sync driver for GraphRAG components  
        >>> with client.sync_driver.session() as session:
        ...     result = session.run("RETURN 1")
    
    Note:
        Both drivers connect to the same Neo4j database but serve different
        purposes: async for web requests, sync for GraphRAG processing.
    """
    
    def __init__(self):
        # Async driver for FastAPI endpoints
        self.async_driver: Optional[AsyncDriver] = None
        
        # Sync driver for Neo4j GraphRAG components  
        self.sync_driver: Optional[Driver] = None
        
        # Separate locks for thread-safe initialization
        self._async_lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()
    
    async def connect_async(self) -> None:
        """
        Establish asynchronous connection for FastAPI endpoints.
        
        Creates an async Neo4j driver optimized for FastAPI web requests
        with appropriate connection pooling for concurrent HTTP operations.
        
        Raises:
            ConnectionError: If unable to connect to Neo4j database
            AuthError: If authentication credentials are invalid
            
        Note:
            This method is idempotent - calling it multiple times will
            reuse the existing connection if already established.
        """
        if self.async_driver is None:
            async with self._async_lock:
                if self.async_driver is None:
                    try:
                        self.async_driver = AsyncGraphDatabase.driver(
                            settings.NEO4J_URI,
                            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                            max_connection_pool_size=getattr(settings, 'NEO4J_FASTAPI_POOL_SIZE', 100),
                            connection_acquisition_timeout=getattr(settings, 'NEO4J_FASTAPI_TIMEOUT', 30)
                        )
                        # Test connection
                        await self.async_driver.verify_connectivity()
                        logger.info("FastAPI async driver connected successfully")
                        
                        # Create indexes on first async connection
                        await self._create_initial_indexes()
                        
                    except Exception as e:
                        logger.error(f"Failed to create async driver: {e}")
                        raise ConnectionError(f"Cannot establish async Neo4j connection: {e}") from e

    def connect_sync(self) -> None:
        """
        Establish synchronous connection for neo4j_graphrag components.
        
        Creates a sync Neo4j driver specifically for GraphRAG components
        (VectorRetriever, Neo4jWriter, etc.) which require synchronous drivers.
        
        Raises:
            ConnectionError: If unable to connect to Neo4j database
            AuthError: If authentication credentials are invalid
            
        Note:
            GraphRAG components require synchronous drivers. This method
            creates a separate connection pool from the async driver.
        """
        if self.sync_driver is None:
            # Note: Using a simple check here since sync operations don't need async locks
            try:
                self.sync_driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                    max_connection_pool_size=getattr(settings, 'NEO4J_WORKER_POOL_SIZE', 50),
                    connection_acquisition_timeout=getattr(settings, 'NEO4J_WORKER_TIMEOUT', 30)
                )
                # Test connection
                self.sync_driver.verify_connectivity()
                logger.info("GraphRAG sync driver connected successfully")
                
            except Exception as e:
                logger.error(f"Failed to create sync driver: {e}")
                raise ConnectionError(f"Cannot establish sync Neo4j connection: {e}") from e
    
    # Backward compatibility method
    async def connect(self) -> None:
        """
        Establish connection to Neo4j (backward compatibility).
        
        For backward compatibility with existing code that calls connect().
        This method establishes the async connection.
        
        Note:
            Use connect_async() for new code to be explicit about driver type.
        """
        await self.connect_async()
    
    async def _create_initial_indexes(self):
        """Create indexes for graph isolation"""
        try:
            # Check if indexes exist first to avoid notifications
            index_configs = [
                ("graph_id_entities", "Entity", "graph_id"),
                ("graph_id_chunks", "Chunk", "graph_id"),
                ("graph_id_documents", "Document", "graph_id")
            ]
            
            # Get existing indexes
            existing_indexes = await self.execute_query("SHOW INDEXES")
            existing_index_names = {record.get("name", "") for record in existing_indexes} if existing_indexes else set()
            
            for index_name, node_label, property_name in index_configs:
                if index_name not in existing_index_names:
                    query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{node_label}) ON (n.{property_name})"
                    try:
                        await self.execute_write_query(query)
                        logger.debug(f"Created index: {index_name}")
                    except Exception as e:
                        logger.debug(f"Failed to create index {index_name}: {e}")
                else:
                    logger.debug(f"Index {index_name} already exists, skipping")
                    
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    async def disconnect(self) -> None:
        """
        Close both async and sync connections.
        
        Cleanly shuts down both driver connections and their associated
        connection pools. Safe to call multiple times.
        
        Note:
            Always call this during application shutdown to ensure
            proper cleanup of connection resources.
        """
        async with self._async_lock:
            if self.async_driver:
                try:
                    await self.async_driver.close()
                    logger.info("Async driver disconnected")
                except Exception as e:
                    logger.warning(f"Error closing async driver: {e}")
                finally:
                    self.async_driver = None
        
        if self.sync_driver:
            try:
                self.sync_driver.close()
                logger.info("Sync driver disconnected")
            except Exception as e:
                logger.warning(f"Error closing sync driver: {e}")
            finally:
                self.sync_driver = None

    # Backward compatibility method
    async def close(self) -> None:
        """
        Close connection to Neo4j (backward compatibility).
        
        For backward compatibility with existing code that calls close().
        This method disconnects both drivers.
        
        Note:
            Use disconnect() for new code to be explicit about both drivers.
        """
        await self.disconnect()
    
    async def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None  # Ignored - always use default
    ) -> List[Dict[str, Any]]:
        """
        Execute a read query using the async driver.
        
        Args:
            query: Cypher query string
            parameters: Optional query parameters
            database: Ignored for compatibility (always uses configured database)
            
        Returns:
            List of query result records as dictionaries
            
        Raises:
            ConnectionError: If async driver is not available
            
        Note:
            Automatically connects async driver if not already connected.
        """
        if not self.async_driver:
            await self.connect_async()

        # Sanitize query for span attribute (first 200 chars, no parameter values)
        query_summary = query.strip()[:200]
        with _tracer.start_as_current_span(
            "neo4j.query",
            kind=otel_trace.SpanKind.CLIENT,
        ) as span:
            span.set_attribute("db.system", "neo4j")
            span.set_attribute("db.operation", "read")
            span.set_attribute("db.statement", query_summary)
            try:
                async with self.async_driver.session(database=settings.NEO4J_DATABASE) as session:
                    result = await session.run(query, parameters or {})
                    records = await result.data()
                    span.set_attribute("db.neo4j.row_count", len(records))
                    return records
            except Exception as e:
                span.record_exception(e)
                span.set_status(otel_trace.StatusCode.ERROR, str(e))
                logger.error(f"Query execution failed: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Parameters: {parameters}")
                raise
    
    async def execute_write_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None  # Ignored - always use default
    ) -> List[Dict[str, Any]]:
        """
        Execute a write query using the async driver.
        
        Args:
            query: Cypher write query string
            parameters: Optional query parameters
            database: Ignored for compatibility (always uses configured database)
            
        Returns:
            List of query result records as dictionaries
            
        Raises:
            ConnectionError: If async driver is not available
            
        Note:
            Uses write transactions for data consistency.
            Automatically connects async driver if not already connected.
        """
        if not self.async_driver:
            await self.connect_async()

        query_summary = query.strip()[:200]
        with _tracer.start_as_current_span(
            "neo4j.write_query",
            kind=otel_trace.SpanKind.CLIENT,
        ) as span:
            span.set_attribute("db.system", "neo4j")
            span.set_attribute("db.operation", "write")
            span.set_attribute("db.statement", query_summary)
            try:
                async with self.async_driver.session(database=settings.NEO4J_DATABASE) as session:
                    async def tx_func(tx):
                        result = await tx.run(query, parameters or {})
                        return await result.data()

                    result = await session.execute_write(tx_func)
                    span.set_attribute("db.neo4j.row_count", len(result))
                    return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(otel_trace.StatusCode.ERROR, str(e))
                logger.error(f"Write query execution failed: {e}")
                raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Neo4j connection health for both drivers.
        
        Returns:
            Dictionary with health status information including
            connectivity status for both async and sync drivers.
        """
        try:
            # Test async driver connection
            if not self.async_driver:
                await self.connect_async()
            
            # Test basic connectivity
            await self.execute_query("RETURN 1 as health")
            
            # Get database info
            db_info = await self.execute_query(
                "CALL dbms.components() YIELD name, versions, edition"
            )
            
            # Test sync driver if needed by GraphRAG components
            sync_status = "not_initialized"
            if self.sync_driver:
                try:
                    with self.sync_driver.session(database=settings.NEO4J_DATABASE) as session:
                        session.run("RETURN 1").consume()
                    sync_status = "healthy"
                except Exception:
                    sync_status = "unhealthy"
            
            return {
                "status": "healthy",
                "async_driver": "connected",
                "sync_driver": sync_status,
                "database_info": db_info[0] if db_info else None,
                "uri": settings.NEO4J_URI,
                "database": settings.NEO4J_DATABASE
            }
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return {
                "status": "unhealthy",
                "async_driver": "error", 
                "sync_driver": "unknown",
                "error": str(e),
                "uri": settings.NEO4J_URI
            }

# Global client instance
neo4j_client = Neo4jClient()
