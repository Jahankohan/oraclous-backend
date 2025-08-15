"""
Neo4j Connection Pool - Async connection pool management for Neo4j
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, AsyncGenerator
import logging
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from app.config.settings import get_settings
from app.core.exceptions import Neo4jConnectionError

logger = logging.getLogger(__name__)


class Neo4jPool:
    """Async Neo4j connection pool manager"""
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: float = 60.0,
        max_transaction_retry_time: float = 30.0
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver: Optional[AsyncDriver] = None
        self._config = {
            "max_connection_pool_size": max_connection_pool_size,
            "connection_acquisition_timeout": connection_acquisition_timeout,
            "max_transaction_retry_time": max_transaction_retry_time,
            "user_agent": get_settings().neo4j_user_agent
        }
        self._lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Initialize the connection pool"""
        async with self._lock:
            if self._driver is not None:
                return
            
            try:
                self._driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    **self._config
                )
                
                # Verify connectivity
                async with self._driver.session() as session:
                    await session.run("RETURN 1")
                
                logger.info(f"Connected to Neo4j at {self.uri}")
                
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise Neo4jConnectionError(f"Cannot connect to Neo4j: {e}")
    
    async def close(self) -> None:
        """Close the connection pool"""
        async with self._lock:
            if self._driver:
                await self._driver.close()
                self._driver = None
                logger.info("Neo4j connection pool closed")
    
    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[AsyncSession, None]:
        """Acquire a session from the pool"""
        if not self._driver:
            await self.connect()
        
        session = None
        try:
            session = self._driver.session(database=self.database)
            yield session
        finally:
            if session:
                await session.close()
    
    async def execute_read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        transformer: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Execute a read query with automatic retry"""
        async with self.acquire() as session:
            try:
                result = await session.execute_read(
                    self._execute_query,
                    query,
                    parameters or {},
                    transformer
                )
                return result
            except (ServiceUnavailable, SessionExpired) as e:
                logger.warning(f"Retrying query due to: {e}")
                # Retry once more
                result = await session.execute_read(
                    self._execute_query,
                    query,
                    parameters or {},
                    transformer
                )
                return result
    
    async def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        transformer: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Execute a write query with automatic retry"""
        async with self.acquire() as session:
            try:
                result = await session.execute_write(
                    self._execute_query,
                    query,
                    parameters or {},
                    transformer
                )
                return result
            except (ServiceUnavailable, SessionExpired) as e:
                logger.warning(f"Retrying query due to: {e}")
                # Retry once more
                result = await session.execute_write(
                    self._execute_query,
                    query,
                    parameters or {},
                    transformer
                )
                return result
    
    @staticmethod
    async def _execute_query(
        tx,
        query: str,
        parameters: Dict[str, Any],
        transformer: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query within a transaction"""
        result = await tx.run(query, parameters)
        records = await result.data()
        
        if transformer:
            return [transformer(record) for record in records]
        return records
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get database schema information"""
        schema_queries = {
            "node_labels": "CALL db.labels()",
            "relationship_types": "CALL db.relationshipTypes()",
            "property_keys": "CALL db.propertyKeys()",
            "indexes": "SHOW INDEXES",
            "constraints": "SHOW CONSTRAINTS"
        }
        
        schema = {}
        for key, query in schema_queries.items():
            try:
                result = await self.execute_read(query)
                schema[key] = result
            except Exception as e:
                logger.warning(f"Failed to get {key}: {e}")
                schema[key] = []
        
        return schema
    
    async def check_vector_index_dimensions(self) -> Optional[int]:
        """Check existing vector index dimensions"""
        query = """
        SHOW INDEXES 
        WHERE type = 'VECTOR'
        RETURN name, options
        """
        try:
            result = await self.execute_read(query)
            if result:
                options = result[0].get('options', {})
                return options.get('indexConfig', {}).get('vector.dimensions')
        except Exception as e:
            logger.warning(f"Failed to check vector dimensions: {e}")
        return None
    
    async def create_vector_index(
        self,
        index_name: str = "vector",
        label: str = "Chunk",
        property_name: str = "embedding",
        dimensions: int = 384
    ) -> None:
        """Create vector index"""
        query = f"""
        CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
        FOR (n:`{label}`) ON (n.`{property_name}`)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        await self.execute_write(query)
        logger.info(f"Vector index '{index_name}' created/updated")
    
    async def drop_vector_index(self, index_name: str = "vector") -> None:
        """Drop vector index"""
        query = f"DROP INDEX `{index_name}` IF EXISTS"
        await self.execute_write(query)
        logger.info(f"Vector index '{index_name}' dropped")
    
    async def create_fulltext_index(
    self,
    index_name: str = "chunk_fulltext",
    labels: List[str] = None,
    properties: List[str] = None,
    options: dict = None
    ) -> None:
        """Create a fulltext index using Neo4j 5.x syntax."""
        labels = labels or ["Chunk"]
        properties = properties or ["text"]
        options = options or {}

        label_str = "|".join([f"`{label}`" for label in labels])
        prop_str = ", ".join([f"n.`{prop}`" for prop in properties])

        opts_str = ""
        if options:
            # Format options map for query (keys and values as strings)
            cfg = ", ".join([f"`{k}`: '{v}'" for k, v in options.items()])
            opts_str = f"\nOPTIONS {{ {cfg} }}"

        query = f"""
        CREATE FULLTEXT INDEX `{index_name}` IF NOT EXISTS
        FOR (n:{label_str})
        ON EACH [{prop_str}]{opts_str}
        """

        await self.execute_write(query)
        logger.info(f"Fulltext index '{index_name}' created")
    
    async def ensure_constraints(self) -> None:
        """Ensure necessary constraints exist"""
        constraints = [
            ("Document", "id"),
            ("Chunk", "id"),
            ("Entity", "id"),
            ("Session", "id"),
            ("Message", "id")
        ]
        
        for label, property_name in constraints:
            try:
                query = f"""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (n:{label})
                REQUIRE n.{property_name} IS UNIQUE
                """
                await self.execute_write(query)
                logger.info(f"Ensured constraint for {label}.{property_name}")
            except Exception as e:
                logger.warning(f"Failed to create constraint for {label}.{property_name}: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the database"""
        try:
            async with self.acquire() as session:
                result = await session.run("RETURN 1 as status")
                record = await result.single()
                
                if record and record["status"] == 1:
                    # Get additional metrics
                    metrics_query = """
                    MATCH (n)
                    WITH labels(n) as label
                    RETURN label, count(*) as count
                    ORDER BY count DESC
                    LIMIT 10
                    """
                    
                    metrics = await self.execute_read(metrics_query)
                    
                    return {
                        "status": "healthy",
                        "connected": True,
                        "database": self.database,
                        "node_counts": {m["label"][0]: m["count"] for m in metrics if m["label"]}
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "connected": False,
                        "error": "Failed to execute test query"
                    }
                    
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }


# Global pool instance
_pool: Optional[Neo4jPool] = None


async def get_neo4j_pool() -> Neo4jPool:
    """Get or create the global Neo4j pool"""
    global _pool
    
    if _pool is None:
        settings = get_settings()
        _pool = Neo4jPool(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database
        )
        await _pool.connect()
    
    return _pool


async def close_neo4j_pool() -> None:
    """Close the global Neo4j pool"""
    global _pool
    
    if _pool:
        await _pool.close()
        _pool = None