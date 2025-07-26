from fastmcp import FastMCP

from pydantic import BaseModel
from typing import List

mcp = FastMCP(name="PostgresWriterMock", host="0.0.0.0", port=8080)

@mcp.tool()
def postgres_writer(records: list[dict]) -> dict:
    """
    Mock writing QA records to Postgres.
    """
    return {
        "success": True, 
        "message": f"Successfully stored {len(records)}  of data",
    }

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8080
    )
