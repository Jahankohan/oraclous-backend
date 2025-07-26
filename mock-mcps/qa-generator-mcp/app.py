from fastmcp import FastMCP
from typing import List, Optional
from pydantic import BaseModel

mcp = FastMCP(name="QAGeneratorMock", host="0.0.0.0", port=8080)

@mcp.tool()
def qa_generator(documents: list[dict]) -> list[dict]:
    """
    Generate mock QA pairs from input documents.
    """
    
    return {
        "success": True, 
        "message": "QA pairs generated", 
        "data": [
            {
                "question": "Why refactor the getUser method?",
                "answer": "It handles undefined ID without a null check.",
                "metadata": {"source": "github"}
            },
            {
                "question": "What is missing in validateUser?",
                "answer": "Error handling for null tokens.",
                "metadata": {"source": "github"}
            }
        ]
    }

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8080
    )
