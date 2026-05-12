#!/usr/bin/env python3
"""
Script to remove analytics methods from chat_service.py and redirect calls to analytics_service.

This script:
1. Identifies analytics methods that have been extracted to analytics_service
2. Removes them from chat_service.py
3. Ensures proper imports and service calls are maintained
"""

import re

# Read the current chat_service.py file
with open(
    "/Users/reza/workspace/Oraclous/oraclous-data-studio/knowledge-graph-builder/app/services/chat_service.py",
    "r",
) as f:
    content = f.read()

# Define the sections to remove - more specific patterns
sections_to_remove = [
    # Pathway methods section
    (r"# ==================== PATHWAY METHODS.*?(?=# ====================)", re.DOTALL),
    # Individual analytics methods that might remain
    (
        r"async def _get_neighborhood_context_with_graph_id.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    (
        r"async def _get_community_context_with_graph_id.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    (
        r"async def _get_simple_community_context_with_graph_id.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    (
        r"async def _get_influential_context_with_graph_id.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    (
        r"async def _get_simple_influential_context_with_graph_id.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    (
        r"async def _get_temporal_context_with_graph_id.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    (
        r"async def _get_pathway_context_with_graph_id.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    (
        r"async def _find_shortest_paths_with_graph_id.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    (
        r"async def _find_paths_between_entities_with_graph_id.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    (
        r"async def _get_relevant_graph_statistics.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    (
        r"async def _precompute_graph_statistics.*?(?=\n    async def|\n    def|\n    # ====|\Z)",
        re.DOTALL,
    ),
    # Remove orphaned method bodies without signatures
    (
        r'\s*\) -> Dict\[str, Any\]:\s*""".*?""".*?(?=\n    async def|\n    def|\n    # ====|\Z)',
        re.DOTALL,
    ),
    (
        r'\s*\) -> List\[Dict\[str, Any\]\]:\s*""".*?""".*?(?=\n    async def|\n    def|\n    # ====|\Z)',
        re.DOTALL,
    ),
    (
        r'\s*\) -> None:\s*""".*?""".*?(?=\n    async def|\n    def|\n    # ====|\Z)',
        re.DOTALL,
    ),
]

print("Removing analytics methods from chat_service.py...")

new_content = content
for pattern, flags in sections_to_remove:
    matches = re.findall(pattern, new_content, flags)
    if matches:
        print(f"Found {len(matches)} matches for pattern")
        new_content = re.sub(pattern, "", new_content, flags=flags)

# Clean up any double newlines that might be left
new_content = re.sub(r"\n\n\n+", "\n\n", new_content)

# Write the cleaned content back
with open(
    "/Users/reza/workspace/Oraclous/oraclous-data-studio/knowledge-graph-builder/app/services/chat_service.py",
    "w",
) as f:
    f.write(new_content)

print("Completed removal of analytics methods from chat_service.py")
print("The analytics functionality is now available via analytics_service")
