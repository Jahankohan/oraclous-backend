import os

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://testuser:testpass@postgres:5432/testdatabase"
)
