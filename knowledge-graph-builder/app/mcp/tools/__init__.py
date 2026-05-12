"""MCP tool wrappers for the assessment substrate and registry (TASK-080).

Per ADR-007 (MCP-First), every REST endpoint under `/api/v1/assessments/*`
and `/api/v1/assessments/registry/*` has a 1:1 MCP tool equivalent. The
wrappers in this package call the service layer (`AssessmentService`)
directly — they do NOT go through the HTTP REST surface — so they share
the same boundary, the same JWT plumbing, and the same Pydantic
request/response schemas as the REST endpoints.

Naming convention:
- `assessment.<verb>_<noun>` for assessment-substrate operations
- `registry.<verb>_<noun>` for registry operations

Each module returns a list of `(tool_name, function)` pairs from a
``TOOLS`` constant. ``app.mcp.server`` registers them with FastMCP at
module-import time.
"""

from app.mcp.tools.assessment_tools import TOOLS as ASSESSMENT_TOOLS
from app.mcp.tools.registry_tools import TOOLS as REGISTRY_TOOLS

__all__ = ["ASSESSMENT_TOOLS", "REGISTRY_TOOLS"]
