from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL: str
    ENCRYPTION_KEY: str        
    AUTH_SERVICE_URL: str = "http://auth-service:8000"
    INTERNAL_SERVICE_KEY: str = "THEINTERNALSERVICEKEYISNONSECRETATTHEMOMENT"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()