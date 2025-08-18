from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/oraclous"
    
    # App settings
    APP_NAME: str = "Oraclous Core"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Auth settings (for integration with your existing auth service)
    AUTH_SERVICE_URL: str = "http://localhost:8080"
    CREDENTIAL_BROKER_URL: str = "http://localhost:8001"
    INTERNAL_SERVICE_KEY: str
    
    # Redis (for future job queue)
    REDIS_URL: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"


settings = Settings()

