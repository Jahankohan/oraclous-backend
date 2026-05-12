#!/bin/bash

# Test Runner for Knowledge Graph Builder
# Runs tests in Docker environment with proper service dependencies

set -e

echo "🧪 Knowledge Graph Builder Test Runner"
echo "======================================"

# Default values
TEST_TYPE="all"
VERBOSE=""
CLEANUP=true
SETUP=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        --no-setup)
            SETUP=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --type TYPE       Test type: all, unit, integration, schema, api (default: all)"
            echo "  --verbose, -v     Verbose output"
            echo "  --no-setup        Skip environment setup"
            echo "  --no-cleanup      Skip cleanup after tests"
            echo "  --help, -h        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to wait for service
wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $service to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if docker-compose exec -T $service echo "ready" >/dev/null 2>&1; then
            echo "✅ $service is ready!"
            return 0
        fi

        echo "   Attempt $attempt/$max_attempts - waiting 2 seconds..."
        sleep 2
        ((attempt++))
    done

    echo "❌ Timeout waiting for $service"
    return 1
}

# Setup environment
if [ "$SETUP" = true ]; then
    echo ""
    echo "🚀 Setting up test environment..."

    # Start Docker services
    echo "Starting Docker services..."
    docker-compose up -d

    # Wait for services
    wait_for_service "neo4j"
    wait_for_service "postgres"
    wait_for_service "knowledge-graph-builder"

    # Additional wait for Neo4j initialization
    echo "⏳ Waiting for Neo4j to fully initialize..."
    sleep 10
fi

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add verbosity
if [ -n "$VERBOSE" ]; then
    PYTEST_CMD="$PYTEST_CMD $VERBOSE"
fi

# Add test markers
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD -m unit"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD -m integration"
        ;;
    schema)
        PYTEST_CMD="$PYTEST_CMD -m schema"
        ;;
    api)
        PYTEST_CMD="$PYTEST_CMD -m api"
        ;;
    docker)
        PYTEST_CMD="$PYTEST_CMD -m docker"
        ;;
    all)
        PYTEST_CMD="$PYTEST_CMD --cov=app --cov-report=term-missing"
        ;;
esac

# Add test directory
PYTEST_CMD="$PYTEST_CMD tests/"

echo ""
echo "🧪 Running $TEST_TYPE tests..."
echo "Command: $PYTEST_CMD"
echo ""

# Run tests in Docker
if docker-compose exec -T knowledge-graph-builder $PYTEST_CMD; then
    echo ""
    echo "✅ Tests completed successfully!"
    TEST_SUCCESS=true
else
    echo ""
    echo "❌ Tests failed!"
    TEST_SUCCESS=false
fi

# Cleanup
if [ "$CLEANUP" = true ]; then
    echo ""
    echo "🧹 Cleaning up..."
    docker-compose down -v
    echo "✅ Cleanup complete"
fi

# Exit with appropriate code
if [ "$TEST_SUCCESS" = true ]; then
    echo ""
    echo "🎉 All done! Tests passed."
    exit 0
else
    echo ""
    echo "💥 Tests failed. Check output above for details."
    exit 1
fi
