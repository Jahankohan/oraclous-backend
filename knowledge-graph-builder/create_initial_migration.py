import asyncio
import subprocess
import sys
import os

async def create_migration():
    try:
        # Generate initial migration
        result = subprocess.run([
            sys.executable, "-m", "alembic", "revision", "--autogenerate", 
            "-m", "Initial migration"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Initial migration created successfully")
            print(result.stdout)
        else:
            print("❌ Error creating migration:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(create_migration())
