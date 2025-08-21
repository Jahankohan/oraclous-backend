#!/bin/bash

echo "🔧 Starting Knowledge Graph Builder in development mode..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run setup.sh first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run database migrations
echo "📊 Running database migrations..."
alembic upgrade head

# Start the service
echo "🚀 Starting service on port 8003..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8003 --log-level info
