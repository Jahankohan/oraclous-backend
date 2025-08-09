from typing import Any, Dict, List, Optional
import aiomysql
import pandas as pd

from app.tools.base.database_tool import DatabaseTool
from app.schemas.tool_instance import ExecutionContext, ExecutionResult
from app.schemas.tool_definition import (
    ToolDefinition, ToolSchema, ToolCapability, CredentialRequirement
)
from app.schemas.common import ToolCategory, ToolType, CredentialType


class MySQLReader(DatabaseTool):
    """
    Tool for reading data from MySQL databases
    """
    
    @classmethod
    def get_tool_definition(cls) -> ToolDefinition:
        """Return the tool definition for MySQL Reader"""
        return ToolDefinition(
            name="MySQL Reader",
            description="Execute queries and read data from MySQL databases",
            version="1.0.0",
            category=ToolCategory.INGESTION,
            type=ToolType.INTERNAL,
            capabilities=[
                ToolCapability(
                    name="execute_sql_query",
                    description="Execute SQL queries on MySQL"
                ),
                ToolCapability(
                    name="list_tables",
                    description="List all tables in database"
                ),
                ToolCapability(
                    name="describe_table", 
                    description="Get table schema information"
                ),
                ToolCapability(
                    name="bulk_data_export",
                    description="Export large datasets efficiently"
                )
            ],
            tags=["mysql", "sql", "database", "query", "ingestion"],
            input_schema=ToolSchema(
                type="object",
                properties={
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["query", "list_tables", "describe_table"],
                        "description": "Type of database operation"
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Table name (for describe_table operation)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return"
                    },
                    "parameters": {
                        "type": "array",
                        "description": "Query parameters for parameterized queries"
                    }
                },
                required=["operation"],
                description="MySQL operation parameters"
            ),
            output_schema=ToolSchema(
                type="object",
                properties={
                    "data": {
                        "type": "array",
                        "description": "Query result rows"
                    },
                    "headers": {
                        "type": "array",
                        "description": "Column names"
                    },
                    "row_count": {
                        "type": "integer",
                        "description": "Number of rows returned"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Query execution metadata"
                    }
                },
                description="Database query results"
            ),
            credential_requirements=[
                CredentialRequirement(
                    type=CredentialType.CONNECTION_STRING,
                    required=True,
                    description="MySQL connection string"
                )
            ]
        )
    
    async def _execute_internal(
        self, 
        input_data: Dict[str, Any], 
        context: ExecutionContext
    ) -> ExecutionResult:
        """Execute MySQL operation"""
        try:
            connection_string = self.get_connection_string(context)
            operation = input_data["operation"]
            
            # Parse connection string for aiomysql
            conn_params = self._parse_mysql_connection_string(connection_string)
            
            # Establish connection
            conn = await aiomysql.connect(**conn_params)
            
            try:
                cursor = await conn.cursor()
                
                if operation == "query":
                    result_data = await self._execute_query(cursor, input_data)
                elif operation == "list_tables":
                    result_data = await self._list_tables(cursor)
                elif operation == "describe_table":
                    result_data = await self._describe_table(cursor, input_data)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                
                return ExecutionResult(
                    success=True,
                    data=result_data,
                    metadata={
                        "operation": operation,
                        "database": "mysql"
                    }
                )
                
            finally:
                conn.close()
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=f"MySQL operation failed: {str(e)}"
            )
    
    def _parse_mysql_connection_string(self, connection_string: str) -> Dict[str, Any]:
        """Parse MySQL connection string into aiomysql parameters"""
        # Basic parsing - in production, use proper URL parsing
        # Format: mysql://user:password@host:port/database
        import urllib.parse as urlparse
        
        parsed = urlparse.urlparse(connection_string)
        
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 3306,
            "user": parsed.username,
            "password": parsed.password,
            "db": parsed.path.lstrip("/") if parsed.path else None,
            "autocommit": True
        }
    
    async def _execute_query(
        self, 
        cursor, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute SQL query"""
        query = input_data.get("query")
        if not query:
            raise ValueError("Query is required for query operation")
        
        parameters = input_data.get("parameters", [])
        limit = input_data.get("limit")
        
        # Add LIMIT if specified and not already in query
        if limit and "LIMIT" not in query.upper():
            query = f"{query.rstrip(';')} LIMIT {limit}"
        
        # Execute query
        if parameters:
            await cursor.execute(query, parameters)
        else:
            await cursor.execute(query)
        
        rows = await cursor.fetchall()
        
        if not rows:
            return {
                "data": [],
                "headers": [],
                "row_count": 0,
                "metadata": {"execution_time": 0}
            }
        
        # Get column names
        headers = [desc[0] for desc in cursor.description]
        
        # Convert rows to list of lists
        data = [list(row) for row in rows]
        
        return {
            "data": data,
            "headers": headers,
            "row_count": len(data),
            "metadata": {
                "query_executed": query,
                "parameters_used": bool(parameters)
            }
        }
    
    async def _list_tables(self, cursor) -> Dict[str, Any]:
        """List all tables in the database"""
        await cursor.execute("SHOW TABLES")
        rows = await cursor.fetchall()
        
        headers = ["table_name"]
        data = [list(row) for row in rows]
        
        return {
            "data": data,
            "headers": headers,
            "row_count": len(data),
            "metadata": {"operation": "list_tables"}
        }
    
    async def _describe_table(
        self, 
        cursor, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get table schema information"""
        table_name = input_data.get("table_name")
        if not table_name:
            raise ValueError("table_name is required for describe_table operation")
        
        await cursor.execute(f"DESCRIBE {table_name}")
        rows = await cursor.fetchall()
        
        if not rows:
            raise ValueError(f"Table '{table_name}' not found")
        
        headers = ["field", "type", "null", "key", "default", "extra"]
        data = [list(row) for row in rows]
        
        return {
            "data": data,
            "headers": headers,
            "row_count": len(data),
            "metadata": {
                "operation": "describe_table",
                "table_name": table_name
            }
        }
    
    def calculate_credits(self, input_data: Any, result: ExecutionResult) -> float:
        """Calculate credits based on data processed"""
        if not result.success or not result.data:
            return 0.05  # Minimal charge for failed attempts
        
        row_count = result.data.get("row_count", 0)
        operation = input_data.get("operation", "query")
        
        # Same credit structure as PostgreSQL
        if operation == "list_tables":
            return 0.05
        elif operation == "describe_table":
            return 0.1
        else:  # query operation
            base_credits = 0.1
            row_credits = row_count * 0.001
            return base_credits + row_credits
