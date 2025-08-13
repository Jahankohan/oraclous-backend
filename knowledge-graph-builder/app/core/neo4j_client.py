from neo4j import GraphDatabase, Driver
from typing import Optional, Dict, Any, List
import logging
from contextlib import asynccontextmanager

from app.config.settings import get_settings
from app.core.exceptions import Neo4jConnectionError

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver: Optional[Driver] = None
        
    def connect(self) -> None:
        """Establish connection to Neo4j"""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                user_agent=get_settings().neo4j_user_agent
            )
            # Test connection
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise Neo4jConnectionError(f"Cannot connect to Neo4j: {e}")
    
    def close(self) -> None:
        """Close Neo4j connection"""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query"""
        if not self._driver:
            raise Neo4jConnectionError("No active Neo4j connection")
        
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a write query"""
        if not self._driver:
            raise Neo4jConnectionError("No active Neo4j connection")
        
        try:
            with self._driver.session(database=self.database) as session:
                result = session.execute_write(
                    lambda tx: list(tx.run(query, parameters or {}))
                )
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Write query execution failed: {e}")
            raise
    
    def get_schema(self) -> Dict[str, Any]:
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
                result = self.execute_query(query)
                schema[key] = result
            except Exception as e:
                logger.warning(f"Failed to get {key}: {e}")
                schema[key] = []
        
        return schema
    
    def check_vector_index_dimensions(self) -> Optional[int]:
        """Check existing vector index dimensions"""
        query = """
        SHOW INDEXES 
        WHERE type = 'VECTOR'
        RETURN name, options
        """
        try:
            result = self.execute_query(query)
            if result:
                options = result[0].get('options', {})
                return options.get('indexConfig', {}).get('vector.dimensions')
        except Exception as e:
            logger.warning(f"Failed to check vector dimensions: {e}")
        return None
    
    def create_vector_index(self, index_name: str = "vector", label: str = "Chunk", 
                          property_name: str = "embedding", dimensions: int = 384) -> None:
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
        self.execute_write_query(query)
        logger.info(f"Vector index '{index_name}' created/updated")
    
    def drop_vector_index(self, index_name: str = "vector") -> None:
        """Drop vector index"""
        query = f"DROP INDEX `{index_name}` IF EXISTS"
        self.execute_write_query(query)
        logger.info(f"Vector index '{index_name}' dropped")

# Dependency for FastAPI
def get_neo4j_client() -> Neo4jClient:
    """Get Neo4j client instance"""
    settings = get_settings()
    client = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database
    )
    client.connect()
    try:
        yield client
    finally:
        client.close()
