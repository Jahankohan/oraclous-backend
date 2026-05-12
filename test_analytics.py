#!/usr/bin/env python3

# Simple test to verify analytics service can be imported
try:
    import sys

    sys.path.append(
        "/Users/reza/workspace/Oraclous/oraclous-data-studio/knowledge-graph-builder"
    )

    # Test import of analytics service
    from app.services.analytics_service import analytics_service

    print("✅ Analytics service imported successfully")

    # Test that methods exist
    methods_to_check = [
        "get_community_context",
        "get_neighborhood_context",
        "get_influential_context",
        "get_temporal_context",
        "get_pathway_context",
        "get_graph_statistics",
    ]

    for method in methods_to_check:
        if hasattr(analytics_service, method):
            print(f"✅ Method {method} exists")
        else:
            print(f"❌ Method {method} missing")

except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
