import os
import json

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://testuser:testpass@postgres:5432/testdatabase"
)

MCP_SERVERS=json.loads(
    os.getenv(
        "MCP_SERVERS", ["http://github-mcp:8080/mcp", "http://qa-generator-mcp:8080/mcp", "http://postgres-writer-mcp:8080/mcp"]))

