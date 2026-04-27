"""
Oraclous KGB error code constants.

All structured error codes for the Knowledge Graph Builder service are
defined here.  Import from this module — never hardcode error strings inline.

TASK-030 (rate limiting) imports the PERMISSION_DENIED and other codes from here.
"""

from __future__ import annotations


class KGBError:
    """
    Structured error code constants.

    Each attribute is a 2-tuple: (error_code, message).

    Usage::

        code, message = KGBError.NEO4J_UNAVAILABLE
        return JSONResponse(
            status_code=503,
            content={"error_code": code, "message": message, "retry_after": 30},
            headers={"Retry-After": "30"},
        )
    """

    # ── 5xxx — Infrastructure / service-level errors ──────────────────────────
    NEO4J_UNAVAILABLE = ("KGB-5001", "Graph service temporarily unavailable")
    REDIS_UNAVAILABLE = ("KGB-5002", "Cache service temporarily unavailable")
    LLM_UNAVAILABLE = ("KGB-5003", "LLM service temporarily unavailable")
    CELERY_UNAVAILABLE = ("KGB-5004", "Background job service temporarily unavailable")

    # ── 4xxx — Client / authorization errors ─────────────────────────────────
    GRAPH_NOT_FOUND = ("KGB-4001", "Graph not found")
    PERMISSION_DENIED = ("KGB-4003", "Permission denied")
    RATE_LIMIT_EXCEEDED = ("KGB-4029", "Rate limit exceeded")
