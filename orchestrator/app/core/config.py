import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file before initializing settings
load_dotenv()

# Data source configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIRECTORY", "/app/uploads")

# Supported file types for direct upload
SUPPORTED_FILE_TYPES = {
    "documents": [".pdf", ".txt", ".md", ".docx", ".doc"],
    "data": [".csv", ".json", ".xlsx", ".xls"],
    "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
    "archives": [".zip", ".tar", ".gz", ".rar"]
}

# Data source provider configurations
DATA_SOURCE_PROVIDERS = {
    "google_drive": {
        "display_name": "Google Drive",
        "icon": "google-drive",
        "description": "Access files and folders from Google Drive",
        "auth_provider": "google",
        "default_scopes": ["https://www.googleapis.com/auth/drive.readonly"]
    },
    "google_docs": {
        "display_name": "Google Docs",
        "icon": "google-docs", 
        "description": "Import Google Docs documents",
        "auth_provider": "google",
        "default_scopes": ["https://www.googleapis.com/auth/documents.readonly"]
    },
    "google_sheets": {
        "display_name": "Google Sheets",
        "icon": "google-sheets",
        "description": "Import Google Sheets spreadsheets",
        "auth_provider": "google", 
        "default_scopes": ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    },
    "notion": {
        "display_name": "Notion",
        "icon": "notion",
        "description": "Access Notion pages and databases",
        "auth_provider": "notion",
        "default_scopes": []
    },
    "github": {
        "display_name": "GitHub",
        "icon": "github",
        "description": "Access GitHub repositories and issues",
        "auth_provider": "github",
        "default_scopes": ["repo", "read:user"]
    },
    "file_upload": {
        "display_name": "File Upload",
        "icon": "upload",
        "description": "Upload files directly",
        "auth_provider": None,
        "default_scopes": []
    },
    "database": {
        "display_name": "Database",
        "icon": "database",
        "description": "Connect to SQL databases",
        "auth_provider": None,
        "default_scopes": []
    }
}

# Workflow execution settings
WORKFLOW_TIMEOUT_SECONDS = int(os.getenv("WORKFLOW_TIMEOUT_SECONDS", "3600"))  # 1 hour
MAX_CONCURRENT_WORKFLOWS = int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "10"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT", 
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Cache settings for token management
TOKEN_CACHE_TTL_SECONDS = int(os.getenv("TOKEN_CACHE_TTL_SECONDS", "300"))  # 5 minutes
TOKEN_REFRESH_THRESHOLD_MINUTES = int(os.getenv("TOKEN_REFRESH_THRESHOLD_MINUTES", "30"))

# Rate limiting
API_RATE_LIMIT_PER_MINUTE = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "100"))
PROVIDER_RATE_LIMITS = {
    "google": 100,  # requests per minute
    "notion": 30,   # requests per minute  
    "github": 60    # requests per minute
}

# Security settings
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"

# Feature flags
ENABLE_WEBHOOK_SUPPORT = os.getenv("ENABLE_WEBHOOK_SUPPORT", "true").lower() == "true"
ENABLE_STREAMING_INGESTION = os.getenv("ENABLE_STREAMING_INGESTION", "false").lower() == "true"
ENABLE_DATA_SOURCE_DISCOVERY = os.getenv("ENABLE_DATA_SOURCE_DISCOVERY", "true").lower() == "true"

# Monitoring and health checks
HEALTH_CHECK_INTERVAL_SECONDS = int(os.getenv("HEALTH_CHECK_INTERVAL_SECONDS", "60"))
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"

# Data processing settings
CHUNK_SIZE_BYTES = int(os.getenv("CHUNK_SIZE_BYTES", "1048576"))  # 1MB chunks
MAX_PROCESSING_TIME_SECONDS = int(os.getenv("MAX_PROCESSING_TIME_SECONDS", "1800"))  # 30 minutes
PARALLEL_PROCESSING_WORKERS = int(os.getenv("PARALLEL_PROCESSING_WORKERS", "4"))

# Storage settings
TEMP_STORAGE_PATH = os.getenv("TEMP_STORAGE_PATH", "/tmp/orchestrator")
CLEANUP_TEMP_FILES_AFTER_HOURS = int(os.getenv("CLEANUP_TEMP_FILES_AFTER_HOURS", "24"))

class Settings(BaseSettings):
    DATABASE_URL: str
    AUTH_SERVICE_URL: str = "http://auth-service:8000"
    INTERNAL_SERVICE_KEY: str = "your_internal_service_key"
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
