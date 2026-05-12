"""Pydantic schemas for LLMConfig nodes (STORY-021)."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    anthropic = "anthropic"
    openrouter = "openrouter"
    azure_openai = "azure-openai"


class LLMConfigCreate(BaseModel):
    provider: LLMProvider
    model: str = Field(
        ..., description="Model identifier, e.g. 'claude-sonnet-4-6', 'openai/gpt-4o'"
    )
    api_key: str = Field(
        ...,
        description="Plaintext API key — forwarded to credential-broker, never stored in Neo4j",
    )
    base_url: str | None = Field(
        None, description="Required for openrouter and azure-openai"
    )
    api_version: str | None = Field(
        None, description="Azure OpenAI API version, e.g. '2024-02-01'"
    )


class LLMConfigCreateResponse(BaseModel):
    config_id: str


class LLMConfigResponse(BaseModel):
    config_id: str
    scope: Literal["org", "project"]
    provider: LLMProvider
    model: str
    base_url: str | None
    api_version: str | None
    api_key_masked: str  # "••••" + last 4 chars only
    created_at: int
    deactivated_at: int | None
