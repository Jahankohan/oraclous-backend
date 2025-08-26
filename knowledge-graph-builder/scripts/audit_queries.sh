#!/bin/bash

echo "🔍 Auditing query patterns for multi-tenant violations..."
echo "=================================================="

# Find potentially problematic patterns
echo -e "\n❌ DANGEROUS: Queries without graph_id filtering:"
echo "------------------------------------------------"

# Pattern 1: MATCH without graph_id
grep -rn "MATCH.*{.*id.*}" app/ --include="*.py" | grep -v graph_id | head -10

# Pattern 2: WHERE clauses without graph_id  
echo -e "\n❌ WHERE clauses that might need graph_id:"
grep -rn "WHERE.*\.id.*=" app/ --include="*.py" | grep -v graph_id | head -5

# Pattern 3: Cypher queries in strings
echo -e "\n❌ Raw Cypher strings to review:"
grep -rn '"MATCH\|"""MATCH\|\'MATCH' app/ --include="*.py" | head -5

echo -e "\n✅ GOOD: Queries already using graph_id:"
echo "----------------------------------------"
grep -rn "graph_id.*=" app/ --include="*.py" | head -5

echo -e "\nℹ️ Use this to review and fix each file systematically."
