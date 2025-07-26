from pydantic import BaseModel
from fastmcp import FastMCP

mcp = FastMCP(name="GitHubMock")


@mcp.tool()
def github_ingest(repo_url: str) -> list[dict]:
    """
    Fetch mock GitHub PR comments given a repo URL.
    """
    return {
        "success": True, 
        "message": "Github PR comments fetched successfully.", 
        "data":  
        [ 
            {"source": "github", "content": "Please refactor the getUser method."},
            {"source": "github", "content": "ValidateUser missing error handling."}
        ]
    }

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8080
    )
