#!/bin/bash

echo "🧪 Running tests for Knowledge Graph Builder..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# Run tests with coverage
pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

echo "✅ Tests completed. Coverage report generated in htmlcov/"
