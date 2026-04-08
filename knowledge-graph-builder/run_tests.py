"""
Docker Test Runner

Script to run tests in Docker environment with proper service dependencies.
Handles test database setup, service initialization, and test execution.
"""
import subprocess
import sys
import time
import os
from typing import List, Optional


def run_command(cmd: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a command and handle output."""
    print(f"Running: {' '.join(cmd)}")
    
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        if capture_output and result.stderr:
            print(f"Error: {result.stderr}")
    
    return result


def wait_for_service(service_name: str, max_retries: int = 30) -> bool:
    """Wait for a Docker service to be healthy."""
    print(f"Waiting for {service_name} to be ready...")
    
    for i in range(max_retries):
        try:
            # Check if service is running
            result = run_command([
                "docker-compose", "exec", "-T", service_name, 
                "echo", "Service ready"
            ], capture_output=True)
            
            if result.returncode == 0:
                print(f"{service_name} is ready!")
                return True
                
        except Exception as e:
            print(f"Error checking {service_name}: {e}")
        
        print(f"Attempt {i+1}/{max_retries} - waiting 2 seconds...")
        time.sleep(2)
    
    print(f"Timeout waiting for {service_name}")
    return False


def setup_test_environment():
    """Set up the test environment with Docker services."""
    print("Setting up test environment...")
    
    # Start services in background
    print("Starting Docker services...")
    result = run_command(["docker-compose", "up", "-d"])
    if result.returncode != 0:
        print("Failed to start Docker services")
        return False
    
    # Wait for critical services
    services_to_wait = ["neo4j", "postgres", "knowledge-graph-builder"]
    
    for service in services_to_wait:
        if not wait_for_service(service):
            print(f"Failed to start {service}")
            return False
    
    # Additional wait for Neo4j to be fully ready
    print("Waiting for Neo4j to be fully initialized...")
    time.sleep(10)
    
    return True


def run_tests(test_type: str = "all", test_path: Optional[str] = None, verbose: bool = False) -> bool:
    """Run tests in Docker environment."""
    print(f"Running {test_type} tests...")
    
    # Base pytest command
    pytest_cmd = ["docker-compose", "exec", "-T", "knowledge-graph-builder", "python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        pytest_cmd.append("-v")
    
    # Add test markers
    if test_type == "unit":
        pytest_cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        pytest_cmd.extend(["-m", "integration"])
    elif test_type == "schema":
        pytest_cmd.extend(["-m", "schema"])
    elif test_type == "api":
        pytest_cmd.extend(["-m", "api"])
    elif test_type == "docker":
        pytest_cmd.extend(["-m", "docker"])
    
    # Add specific test path if provided
    if test_path:
        pytest_cmd.append(test_path)
    else:
        pytest_cmd.append("tests/")
    
    # Add coverage if running all tests
    if test_type == "all":
        pytest_cmd.extend(["--cov=app", "--cov-report=term-missing"])
    
    # Run tests
    result = run_command(pytest_cmd)
    
    return result.returncode == 0


def cleanup_test_environment():
    """Clean up test environment."""
    print("Cleaning up test environment...")
    
    # Stop and remove containers
    run_command(["docker-compose", "down", "-v"])
    
    print("Cleanup complete")


def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests in Docker environment")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "schema", "api", "docker"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--path",
        help="Specific test path to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-setup",
        action="store_true", 
        help="Skip environment setup (assume services are running)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip cleanup after tests"
    )
    
    args = parser.parse_args()
    
    success = True
    
    try:
        # Setup environment if requested
        if not args.no_setup:
            if not setup_test_environment():
                print("Failed to set up test environment")
                return 1
        
        # Run tests
        if not run_tests(args.type, args.path, args.verbose):
            print("Tests failed")
            success = False
        
    except KeyboardInterrupt:
        print("\nTest run interrupted")
        success = False
        
    except Exception as e:
        print(f"Error during test run: {e}")
        success = False
        
    finally:
        # Cleanup if requested
        if not args.no_cleanup:
            cleanup_test_environment()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
