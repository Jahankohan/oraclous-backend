"""
Script to test tool implementations
"""

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools.factory import ToolFactory
from app.tools.registry import tool_registry


async def test_tool_execution():
    """Test basic tool execution"""
    print("Testing tool execution...")

    # Test PostgreSQL Reader (without actual database)
    try:
        postgres_definition = None
        for definition in tool_registry.list_definitions():
            if definition.name == "PostgreSQL Reader":
                postgres_definition = definition
                break

        if postgres_definition:
            print(f"\n✓ Found tool: {postgres_definition.name}")
            print(f"  Description: {postgres_definition.description}")
            print(
                f"  Capabilities: {[cap.name for cap in postgres_definition.capabilities]}"
            )

            # Create executor
            executor = ToolFactory.create_executor(postgres_definition.id)
            print(f"✓ Created executor: {executor.__class__.__name__}")

    except Exception as e:
        print(f"✗ Error testing PostgreSQL tool: {e}")

    # Test all registered tools
    print(f"\n📋 All registered tools ({len(tool_registry.list_definitions())}):")
    for i, definition in enumerate(tool_registry.list_definitions(), 1):
        print(f"  {i}. {definition.name} ({definition.category.value})")
        print(f"     Type: {definition.type.value}")
        print(f"     Capabilities: {len(definition.capabilities)}")
        print(f"     Credentials required: {len(definition.credential_requirements)}")


def test_tool_definitions():
    """Test tool definition validation"""
    print("Testing tool definitions...")

    for definition in tool_registry.list_definitions():
        try:
            # Basic validation
            assert definition.name
            assert definition.description
            assert definition.input_schema
            assert definition.output_schema
            assert definition.category
            assert definition.type

            print(f"✓ {definition.name}: Valid definition")

        except Exception as e:
            print(f"✗ {definition.name}: Invalid definition - {e}")


if __name__ == "__main__":
    # Import tools to register them
    from app.tools import tool_registry

    print("=== Tool Definition Tests ===")
    test_tool_definitions()

    print("\n=== Tool Execution Tests ===")
    asyncio.run(test_tool_execution())
