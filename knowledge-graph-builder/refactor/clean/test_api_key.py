#!/usr/bin/env python3
"""Test script to ensure API key is properly passed to OpenAI components.

Reads OPENAI_API_KEY from the environment and verifies both the OpenAI client
and the neo4j-graphrag OpenAIEmbeddings component can authenticate with it.

Fail-closed: raises KeyError at boot if OPENAI_API_KEY is not set.
"""

import os

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI

api_key = os.environ["OPENAI_API_KEY"]

print(f"Testing with API key ending: ...{api_key[-10:]}")

# Test direct OpenAI client
try:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input="test", model="text-embedding-3-large")
    print("✅ Direct OpenAI client works!")
    print(f"Embedding dimension: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"❌ Direct OpenAI client failed: {e}")

# Test neo4j-graphrag embeddings component
try:
    embedder = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    embeddings = embedder.embed_query("test")
    print("✅ neo4j-graphrag OpenAI embeddings works!")
    print(f"Embedding dimension: {len(embeddings)}")
except Exception as e:
    print(f"❌ neo4j-graphrag OpenAI embeddings failed: {e}")
