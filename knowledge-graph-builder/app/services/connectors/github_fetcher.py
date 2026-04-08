"""GitHub connector fetcher (ORA-78, Priority 1)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from app.core.logging import get_logger
from app.services.connectors.base import ConnectorFetcher

logger = get_logger(__name__)

# GitHub GraphQL endpoint for Issues + PRs (supports cursor pagination)
_GITHUB_API = "https://api.github.com"


class GitHubFetcher(ConnectorFetcher):
    """
    Fetches GitHub repository data via the REST API v3.

    Sync targets (per spec §5):
    - /repos/{owner}/{repo}/issues (includes PRs via `pull_request` key)
    - /repos/{owner}/{repo}/commits
    - /repos/{owner}/{repo} (metadata)

    Incremental via `since` parameter (ISO-8601 timestamp cursor).
    """

    def fetch_since(self, cursor: Optional[Any]) -> List[Dict[str, Any]]:
        """
        Fetch issues, PRs, and commits from the configured repository.

        cursor is the ISO-8601 timestamp of the last-synced item.
        """
        repo_owner = self.config.get("repo_owner", "")
        repo_name = self.config.get("repo_name", "")
        if not repo_owner or not repo_name:
            logger.warning("GitHubFetcher: repo_owner and repo_name must be set in config")
            return []

        headers = self._auth_headers()
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"

        items: List[Dict[str, Any]] = []
        new_cursor: Optional[str] = None

        # Fetch issues (& PRs)
        issues = self._fetch_paginated(
            f"{_GITHUB_API}/repos/{repo_owner}/{repo_name}/issues",
            headers=headers,
            params={"state": "all", "per_page": 100, "since": cursor} if cursor else {"state": "all", "per_page": 100},
        )
        for issue in issues:
            issue["_source"] = "issue"
            issue["_repo"] = f"{repo_owner}/{repo_name}"
            items.append(issue)
            # Track latest updated_at as cursor
            updated_at = issue.get("updated_at")
            if updated_at and (not new_cursor or updated_at > new_cursor):
                new_cursor = updated_at

        # Fetch commits
        commits = self._fetch_paginated(
            f"{_GITHUB_API}/repos/{repo_owner}/{repo_name}/commits",
            headers=headers,
            params={"per_page": 100, "since": cursor} if cursor else {"per_page": 100},
        )
        for commit in commits:
            commit["_source"] = "commit"
            commit["_repo"] = f"{repo_owner}/{repo_name}"
            items.append(commit)

        self.last_cursor = new_cursor
        logger.info(f"GitHubFetcher: fetched {len(items)} items from {repo_owner}/{repo_name}")
        return items

    def to_text(self, items: List[Dict[str, Any]]) -> str:
        """Convert GitHub items to human-readable text for LLM extraction."""
        parts: List[str] = []
        for item in items:
            source = item.get("_source", "item")
            repo = item.get("_repo", "")

            if source == "issue":
                is_pr = "pull_request" in item
                kind = "Pull Request" if is_pr else "Issue"
                number = item.get("number", "")
                title = item.get("title", "")
                body = (item.get("body") or "")[:500]
                user = (item.get("user") or {}).get("login", "unknown")
                state = item.get("state", "")
                labels = ", ".join(lbl.get("name", "") for lbl in item.get("labels", []))
                parts.append(
                    f"{kind} #{number} in {repo}: {title}\n"
                    f"Author: {user} | State: {state} | Labels: {labels}\n"
                    f"{body}\n"
                )
            elif source == "commit":
                sha = item.get("sha", "")[:8]
                commit_data = item.get("commit", {})
                message = (commit_data.get("message") or "")[:300]
                author = (commit_data.get("author") or {}).get("name", "unknown")
                parts.append(
                    f"Commit {sha} in {repo} by {author}:\n{message}\n"
                )

        return "\n---\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_paginated(
        self,
        url: str,
        headers: Dict[str, str],
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Fetch all pages from a GitHub REST endpoint."""
        results: List[Dict[str, Any]] = []
        next_url: Optional[str] = url

        with httpx.Client(timeout=30.0) as client:
            while next_url:
                self._rate_limit_wait()
                try:
                    response = client.get(next_url, headers=headers, params=params if next_url == url else None)
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    logger.error(f"GitHubFetcher HTTP error {e.response.status_code}: {e}")
                    break
                except Exception as e:
                    logger.error(f"GitHubFetcher request error: {e}")
                    break

                data = response.json()
                if isinstance(data, list):
                    results.extend(data)
                elif isinstance(data, dict) and "items" in data:
                    results.extend(data["items"])

                # Follow Link header for next page
                next_url = self._parse_next_link(response.headers.get("Link", ""))

        return results

    def _parse_next_link(self, link_header: str) -> Optional[str]:
        """Parse GitHub Link header to get the 'next' page URL."""
        if not link_header:
            return None
        for part in link_header.split(","):
            url_part, *rel_parts = part.strip().split(";")
            if any('rel="next"' in rp for rp in rel_parts):
                return url_part.strip().strip("<>")
        return None
