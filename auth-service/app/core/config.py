from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file before initializing settings
load_dotenv()


class Settings(BaseSettings):
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: str = 1440  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: str = 7 # 7 days
    DB_URL: str
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GITHUB_CLIENT_ID: str
    GITHUB_CLIENT_SECRET: str
    NOTION_CLIENT_ID: str
    NOTION_CLIENT_SECRET: str
    REDIRECT_URI: str
    EMAIL_ADDRESS: str
    EMAIL_PASSWORD: str
    FRONTEND_URL: str = "http://localhost:8080"
    INTERNAL_SERVICE_KEY: str = "your_internal_service_key"
    REDIS_URL: str = "redis://redis:6379"
    # Explicit CORS origins — never use wildcard with allow_credentials=True
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8080"]
    # Trusted reverse-proxy IPs (ALB/nginx). ProxyHeadersMiddleware only reads
    # X-Forwarded-For when the direct connection comes from one of these IPs.
    # Default: loopback only (safe for local dev; configure for production).
    TRUSTED_PROXY_IPS: list[str] = ["127.0.0.1"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
