"""Pydantic schemas for the Connector Framework (ORA-78)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Auth config
# ---------------------------------------------------------------------------


class AuthConfig(BaseModel):
    auth_type: Literal["api_key", "bearer_token", "oauth2", "basic", "hmac_secret"]

    # api_key / bearer_token
    credential_id: str | None = None
    header_name: str | None = "Authorization"

    # oauth2
    token_url: str | None = None
    client_id: str | None = None
    client_secret_credential_id: str | None = None
    scopes: list[str] | None = None

    # hmac (webhook verification)
    hmac_secret_credential_id: str | None = None
    hmac_algorithm: str | None = "sha256"
    hmac_header: str | None = "X-Hub-Signature-256"


# ---------------------------------------------------------------------------
# Pagination config
# ---------------------------------------------------------------------------


class PaginationConfig(BaseModel):
    strategy: Literal["cursor", "offset", "page", "link_header", "none"]
    cursor_field: str | None = None  # e.g. "after", "cursor"
    cursor_path: str | None = None  # JSONPath to cursor in response
    items_path: str = "data"  # JSONPath to items array
    page_param: str | None = "page"
    per_page_param: str | None = "per_page"
    per_page_default: int = 100


# ---------------------------------------------------------------------------
# Entity mapping config
# ---------------------------------------------------------------------------


class EntityMappingConfig(BaseModel):
    extraction_mode: Literal["llm", "template", "passthrough"]
    context_hint: str | None = None  # e.g. "GitHub repository metadata"
    field_mappings: dict[str, str] | None = None  # field → entity type (template mode)


# ---------------------------------------------------------------------------
# Full connector config (stored as JSONB in PostgreSQL)
# ---------------------------------------------------------------------------


class ConnectorConfig(BaseModel):
    auth: AuthConfig
    base_url: str | None = None
    pagination: PaginationConfig | None = None
    entity_mapping: EntityMappingConfig
    incremental: bool = True  # Use cursor for incremental sync
    rate_limit_rps: float | None = None  # Requests per second limit


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

CONNECTOR_TYPES = Literal[
    "github",
    "notion",
    "linear",
    "confluence",
    "slack",
    "rest_api",
    "webhook_receiver",
]


class RegisterConnectorRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    connector_type: CONNECTOR_TYPES
    config: ConnectorConfig
    schedule: str | None = None  # cron expression; None = webhook-only


class UpdateConnectorRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    config: ConnectorConfig | None = None
    schedule: str | None = None
    status: Literal["active", "paused"] | None = None


class ConnectorResponse(BaseModel):
    id: str
    graph_id: str
    name: str
    connector_type: str
    status: str
    schedule: str | None
    last_synced_at: datetime | None
    webhook_url: str | None = None  # populated for webhook_receiver type
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConnectorListResponse(BaseModel):
    connectors: list[ConnectorResponse]
    total: int


# ---------------------------------------------------------------------------
# Sync log schemas
# ---------------------------------------------------------------------------


class SyncLogResponse(BaseModel):
    id: str
    connector_id: str
    started_at: datetime
    finished_at: datetime | None
    status: str | None
    items_processed: int
    entities_extracted: int
    error_message: str | None

    model_config = {"from_attributes": True}


class SyncLogListResponse(BaseModel):
    logs: list[SyncLogResponse]
    total: int


# ---------------------------------------------------------------------------
# Connector template schemas
# ---------------------------------------------------------------------------


class ConnectorTemplate(BaseModel):
    connector_type: str
    display_name: str
    description: str
    default_config: dict[str, Any]
    required_credentials: list[str]
    supports_incremental: bool
    supports_webhook: bool


# ---------------------------------------------------------------------------
# Built-in connector templates (spec §5)
# ---------------------------------------------------------------------------

CONNECTOR_TEMPLATES: dict[str, ConnectorTemplate] = {
    "github": ConnectorTemplate(
        connector_type="github",
        display_name="GitHub",
        description="Sync repositories, issues, pull requests, and contributors into your knowledge graph",
        default_config={
            "auth": {"auth_type": "bearer_token", "header_name": "Authorization"},
            "base_url": "https://api.github.com",
            "pagination": {
                "strategy": "cursor",
                "cursor_field": "after",
                "items_path": "data",
            },
            "entity_mapping": {
                "extraction_mode": "template",
                "context_hint": "GitHub repository: extract Repository, Person (contributors), Issue, PullRequest, Commit entities",
                "field_mappings": {
                    "repository": "Repository",
                    "user.login": "Person",
                    "issue": "Issue",
                    "pull_request": "PullRequest",
                },
            },
            "incremental": True,
        },
        required_credentials=["bearer_token"],
        supports_incremental=True,
        supports_webhook=True,
    ),
    "notion": ConnectorTemplate(
        connector_type="notion",
        display_name="Notion",
        description="Sync pages, databases, and blocks from Notion workspaces",
        default_config={
            "auth": {"auth_type": "bearer_token", "header_name": "Authorization"},
            "base_url": "https://api.notion.com/v1",
            "pagination": {
                "strategy": "cursor",
                "cursor_field": "start_cursor",
                "cursor_path": "next_cursor",
                "items_path": "results",
            },
            "entity_mapping": {
                "extraction_mode": "llm",
                "context_hint": "Notion page or database: extract Document, Concept, Topic, Person entities",
            },
            "incremental": True,
        },
        required_credentials=["bearer_token"],
        supports_incremental=True,
        supports_webhook=True,
    ),
    "linear": ConnectorTemplate(
        connector_type="linear",
        display_name="Linear",
        description="Sync issues, projects, and teams from Linear",
        default_config={
            "auth": {"auth_type": "bearer_token", "header_name": "Authorization"},
            "base_url": "https://api.linear.app/graphql",
            "pagination": {
                "strategy": "cursor",
                "cursor_field": "after",
                "items_path": "nodes",
            },
            "entity_mapping": {
                "extraction_mode": "template",
                "context_hint": "Linear engineering tool: extract Issue, Project, Person, Team entities",
                "field_mappings": {
                    "issue": "Issue",
                    "project": "Project",
                    "user": "Person",
                    "team": "Team",
                },
            },
            "incremental": True,
        },
        required_credentials=["bearer_token"],
        supports_incremental=True,
        supports_webhook=True,
    ),
    "confluence": ConnectorTemplate(
        connector_type="confluence",
        display_name="Confluence",
        description="Sync spaces and pages from Confluence for enterprise documentation graphs",
        default_config={
            "auth": {"auth_type": "basic"},
            "pagination": {
                "strategy": "offset",
                "page_param": "start",
                "per_page_param": "limit",
                "items_path": "results",
            },
            "entity_mapping": {
                "extraction_mode": "llm",
                "context_hint": "Confluence documentation: extract Document, Topic, Person entities",
            },
            "incremental": True,
        },
        required_credentials=["basic_credentials"],
        supports_incremental=True,
        supports_webhook=False,
    ),
    "slack": ConnectorTemplate(
        connector_type="slack",
        display_name="Slack",
        description="Sync channel history and user directory from Slack workspaces",
        default_config={
            "auth": {"auth_type": "bearer_token", "header_name": "Authorization"},
            "base_url": "https://slack.com/api",
            "pagination": {
                "strategy": "cursor",
                "cursor_path": "response_metadata.next_cursor",
                "items_path": "messages",
            },
            "entity_mapping": {
                "extraction_mode": "llm",
                "context_hint": "Slack workspace: extract Person, Conversation, Topic entities",
            },
            "incremental": True,
            "rate_limit_rps": 0.8,
        },
        required_credentials=["bearer_token"],
        supports_incremental=True,
        supports_webhook=True,
    ),
    "rest_api": ConnectorTemplate(
        connector_type="rest_api",
        display_name="Generic REST API",
        description="Connect any JSON REST API — configure endpoint, auth, and pagination",
        default_config={
            "auth": {"auth_type": "api_key"},
            "pagination": {"strategy": "none"},
            "entity_mapping": {"extraction_mode": "llm"},
            "incremental": False,
        },
        required_credentials=["api_key_or_token"],
        supports_incremental=False,
        supports_webhook=False,
    ),
    "webhook_receiver": ConnectorTemplate(
        connector_type="webhook_receiver",
        display_name="Webhook Receiver",
        description="Receive real-time push events from external services via HMAC-signed webhooks",
        default_config={
            "auth": {
                "auth_type": "hmac_secret",
                "hmac_algorithm": "sha256",
                "hmac_header": "X-Hub-Signature-256",
            },
            "entity_mapping": {"extraction_mode": "llm"},
        },
        required_credentials=["hmac_secret"],
        supports_incremental=False,
        supports_webhook=True,
    ),
}
