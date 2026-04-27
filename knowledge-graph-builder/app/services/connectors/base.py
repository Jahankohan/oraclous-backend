"""Base connector fetcher interface (ORA-78)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class ConnectorFetcher(ABC):
    """
    Abstract base class for all connector fetchers.

    Each subclass handles:
    - Authentication header injection
    - Paginated data fetching
    - Cursor tracking for incremental sync
    - Conversion of raw API items to text for pipeline ingestion
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.last_cursor: Any | None = None
        self._rate_limit_rps: float | None = config.get("rate_limit_rps")
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_since(self, cursor: Any | None) -> list[dict[str, Any]]:
        """
        Fetch items since the given cursor position.

        For incremental connectors, cursor tracks the last-seen item.
        For full-snapshot connectors, cursor is ignored.
        Updates self.last_cursor after fetching.
        """
        ...

    @abstractmethod
    def to_text(self, items: list[dict[str, Any]]) -> str:
        """
        Convert a batch of raw API items to a text string for pipeline ingestion.

        The returned text is fed directly into pipeline_service.process_documents().
        """
        ...

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_type(cls, connector_type: str, config: dict[str, Any]) -> ConnectorFetcher:
        """Return the appropriate fetcher subclass for the given connector_type."""
        from app.services.connectors.github_fetcher import GitHubFetcher
        from app.services.connectors.rest_api_fetcher import RestApiFetcher

        registry: dict[str, type] = {
            "github": GitHubFetcher,
            "rest_api": RestApiFetcher,
            # Additional built-in fetchers can be registered here as implemented.
            # For now, notion/linear/confluence/slack fall back to RestApiFetcher
            # since they share the same paginated REST pattern.
            "notion": RestApiFetcher,
            "linear": RestApiFetcher,
            "confluence": RestApiFetcher,
            "slack": RestApiFetcher,
        }
        fetcher_class = registry.get(connector_type, RestApiFetcher)
        return fetcher_class(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Build auth headers from connector config."""
        auth = self.config.get("auth", {})
        auth_type = auth.get("auth_type", "")
        headers: dict[str, str] = {}

        if auth_type in ("bearer_token", "api_key"):
            credential_value = auth.get("_resolved_credential", "")
            header_name = auth.get("header_name", "Authorization")
            if auth_type == "bearer_token":
                headers[header_name] = f"Bearer {credential_value}"
            else:
                headers[header_name] = credential_value
        elif auth_type == "basic":
            import base64

            username = auth.get("_resolved_username", "")
            password = auth.get("_resolved_password", "")
            encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        return headers

    def _rate_limit_wait(self) -> None:
        """Synchronous sleep to honour rate_limit_rps."""
        if not self._rate_limit_rps:
            return
        min_interval = 1.0 / self._rate_limit_rps
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _chunked(self, items: list[Any], size: int) -> Iterator[list[Any]]:
        """Yield successive chunks from items."""
        for i in range(0, len(items), size):
            yield items[i : i + size]
