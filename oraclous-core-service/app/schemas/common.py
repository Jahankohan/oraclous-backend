from enum import Enum

# Enums for tool categorization
class ToolCategory(str, Enum):
    INGESTION = "INGESTION"
    TRANSFORMERS = "TRANSFORMERS"  
    ANALYTICS = "ANALYTICS"
    STORAGE = "STORAGE"
    FINETUNER = "FINETUNER"


class ToolType(str, Enum):
    INTERNAL = "INTERNAL"  # Implemented in our codebase
    MCP = "MCP"           # Model Context Protocol servers
    API = "API"           # External API integrations


class CredentialType(str, Enum):
    OAUTH_TOKEN = "OAUTH_TOKEN"
    API_KEY = "API_KEY"
    CONNECTION_STRING = "CONNECTION_STRING"
    USERNAME_PASSWORD = "USERNAME_PASSWORD"


class InstanceStatus(str, Enum):
    PENDING = "PENDING"
    CONFIGURATION_REQUIRED = "CONFIGURATION_REQUIRED"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PAUSED = "PAUSED"