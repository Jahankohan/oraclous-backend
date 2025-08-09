#!/bin/bash
# Development setup script

echo "Setting up Oraclous Core development environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file - please update with your configuration"
fi

# Initialize database with Docker
echo "Starting database with Docker..."
docker-compose up -d postgres redis

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 10

# Run database migrations
alembic upgrade head

# Register tools
echo "Registering tools..."
python scripts/register_tools.py

echo "Development setup complete!"
echo "Run 'uvicorn app.main:app --reload' to start the server"
