from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file before initializing settings
load_dotenv()


class Settings(BaseSettings):
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    DB_URL: str
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GITHUB_CLIENT_ID: str
    GITHUB_CLIENT_SECRET: str
    NOTION_CLIENT_ID: str
    NOTION_CLIENT_SECRET: str
    REDIRECT_URI: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
