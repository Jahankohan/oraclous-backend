"""Generic REST API connector fetcher (ORA-78, Priority 6)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.logging import get_logger
from app.services.connectors.base import ConnectorFetcher

logger = get_logger(__name__)


def _get_nested(data: Any, path: str) -> Any:
    """Resolve a dot-notation path in a nested dict/list structure."""
    for key in path.split("."):
        if isinstance(data, dict):
            data = data.get(key)
        elif isinstance(data, list) and key.isdigit():
            data = data[int(key)]
        else:
            return None
    return data


class RestApiFetcher(ConnectorFetcher):
    """
    Generic REST API fetcher.

    Supports cursor, offset, page, link_header, and none pagination strategies
    as defined in PaginationConfig (spec §4.2). Used as fallback for connector
    types that don't have a dedicated fetcher yet (notion, linear, slack, etc.).
    """

    def fetch_since(self, cursor: Any | None) -> list[dict[str, Any]]:
        """Fetch all items from the configured endpoint, using pagination."""
        base_url = self.config.get("base_url", "")
        endpoint = self.config.get("endpoint", "")
        if not base_url and not endpoint:
            logger.warning("RestApiFetcher: base_url or endpoint must be configured")
            return []

        url = (
            endpoint
            if endpoint.startswith("http")
            else f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        )
        headers = self._auth_headers()
        pagination = self.config.get("pagination", {})
        strategy = pagination.get("strategy", "none")
        items_path = pagination.get("items_path", "data")

        all_items: list[dict[str, Any]] = []
        new_cursor: Any | None = cursor

        with httpx.Client(timeout=30.0) as client:
            if strategy == "none":
                all_items = self._fetch_single(client, url, headers, cursor)
                self.last_cursor = None
            elif strategy == "cursor":
                all_items, new_cursor = self._fetch_cursor(
                    client, url, headers, cursor, pagination, items_path
                )
                self.last_cursor = new_cursor
            elif strategy in ("offset", "page"):
                all_items = self._fetch_offset(
                    client, url, headers, pagination, items_path
                )
                self.last_cursor = None
            elif strategy == "link_header":
                all_items = self._fetch_link_header(client, url, headers, items_path)
                self.last_cursor = None

        logger.info(f"RestApiFetcher: fetched {len(all_items)} items from {url}")
        return all_items

    def to_text(self, items: list[dict[str, Any]]) -> str:
        """Convert REST API items to text for LLM extraction."""
        context_hint = (self.config.get("entity_mapping") or {}).get("context_hint", "")
        prefix = f"Context: {context_hint}\n\n" if context_hint else ""
        return prefix + "\n---\n".join(
            json.dumps(item, default=str, ensure_ascii=False) for item in items
        )

    # ------------------------------------------------------------------
    # Pagination strategies
    # ------------------------------------------------------------------

    def _fetch_single(
        self,
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        cursor: Any | None,
    ) -> list[dict[str, Any]]:
        """Single-page fetch — no pagination."""
        self._rate_limit_wait()
        try:
            response = client.get(url, headers=headers)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"RestApiFetcher request error: {e}")
            return []
        data = response.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    def _fetch_cursor(
        self,
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        cursor: Any | None,
        pagination: dict[str, Any],
        items_path: str,
    ) -> tuple[list[dict[str, Any]], Any | None]:
        """Cursor-based pagination."""
        cursor_field = pagination.get("cursor_field", "after")
        cursor_path = pagination.get("cursor_path")
        per_page_param = pagination.get("per_page_param", "per_page")
        per_page = pagination.get("per_page_default", 100)

        all_items: list[dict[str, Any]] = []
        next_cursor = cursor
        params: dict[str, Any] = {per_page_param: per_page}

        while True:
            if next_cursor:
                params[cursor_field] = next_cursor
            self._rate_limit_wait()
            try:
                response = client.get(url, headers=headers, params=params)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"RestApiFetcher cursor request error: {e}")
                break

            data = response.json()
            page_items = _get_nested(data, items_path) or []
            if isinstance(page_items, list):
                all_items.extend(page_items)

            # Get next cursor
            if cursor_path:
                next_cursor = _get_nested(data, cursor_path)
            else:
                next_cursor = None

            if not next_cursor or not page_items:
                break

        return all_items, next_cursor

    def _fetch_offset(
        self,
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        pagination: dict[str, Any],
        items_path: str,
    ) -> list[dict[str, Any]]:
        """Offset/page-based pagination."""
        page_param = pagination.get("page_param", "page")
        per_page_param = pagination.get("per_page_param", "per_page")
        per_page = pagination.get("per_page_default", 100)

        all_items: list[dict[str, Any]] = []
        page = 1

        while True:
            params = {page_param: page, per_page_param: per_page}
            self._rate_limit_wait()
            try:
                response = client.get(url, headers=headers, params=params)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"RestApiFetcher offset request error: {e}")
                break

            data = response.json()
            page_items = _get_nested(data, items_path) or (
                data if isinstance(data, list) else []
            )
            if not page_items:
                break
            all_items.extend(page_items)
            if len(page_items) < per_page:
                break
            page += 1

        return all_items

    def _fetch_link_header(
        self,
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        items_path: str,
    ) -> list[dict[str, Any]]:
        """Link-header-based pagination (RFC 5988)."""
        all_items: list[dict[str, Any]] = []
        next_url: str | None = url

        while next_url:
            self._rate_limit_wait()
            try:
                response = client.get(next_url, headers=headers)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"RestApiFetcher link-header request error: {e}")
                break

            data = response.json()
            page_items = _get_nested(data, items_path) or (
                data if isinstance(data, list) else []
            )
            if isinstance(page_items, list):
                all_items.extend(page_items)

            next_url = self._parse_next_link(response.headers.get("Link", ""))

        return all_items

    def _parse_next_link(self, link_header: str) -> str | None:
        if not link_header:
            return None
        for part in link_header.split(","):
            url_part, *rel_parts = part.strip().split(";")
            if any('rel="next"' in rp for rp in rel_parts):
                return url_part.strip().strip("<>")
        return None
