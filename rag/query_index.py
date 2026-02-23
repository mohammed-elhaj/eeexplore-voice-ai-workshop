"""
RAG: Query the Annoy Vector Index
==================================
Test the RAG system standalone. Loads the pre-built Annoy index,
embeds a query, finds the top 3 nearest chunks, and prints them.

Usage:
  python rag/query_index.py "شنو المسارات المتاحة في EEExtra؟"
  python rag/query_index.py "What topics does Session 1 cover?"
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from annoy import AnnoyIndex

load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
INDEX_DIR = Path(__file__).parent / "index"


def load_index():
    """Load the Annoy index and chunks."""
    index_path = INDEX_DIR / "annoy_index.ann"
    chunks_path = INDEX_DIR / "chunks.json"

    if not index_path.exists():
        print("❌ Index not found! Run: python rag/build_index.py")
        sys.exit(1)

    index = AnnoyIndex(EMBEDDING_DIM, "angular")
    index.load(str(index_path))

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def embed_query(client: genai.Client, query: str) -> list[float]:
    """Embed a query string using Gemini with RETRIEVAL_QUERY task type."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return response.embeddings[0].values


def search(index: AnnoyIndex, chunks: list[dict], query_embedding: list[float], top_k: int = 3):
    """Search the index for nearest neighbors."""
    indices, distances = index.get_nns_by_vector(
        query_embedding, top_k, include_distances=True
    )
    results = []
    for idx, dist in zip(indices, distances):
        results.append({
            "chunk": chunks[idx],
            "distance": dist,
        })
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python rag/query_index.py \"your query here\"")
        print("Example: python rag/query_index.py \"شنو المسارات المتاحة؟\"")
        sys.exit(1)

    query = sys.argv[1]

    print("=" * 60)
    print("🔍 RAG QUERY TEST")
    print("=" * 60)
    print(f"Query: {query}\n")

    # Load index
    index, chunks = load_index()
    print(f"📚 Loaded {len(chunks)} chunks\n")

    # Embed query
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    print("🔢 Embedding query...")
    query_embedding = embed_query(client, query)

    # Search
    print("🔍 Searching for nearest neighbors...\n")
    results = search(index, chunks, query_embedding, top_k=3)

    # Display results
    for i, result in enumerate(results):
        chunk = result["chunk"]
        distance = result["distance"]
        print(f"{'─' * 60}")
        print(f"📄 Result {i + 1}  |  Distance: {distance:.4f}  |  Title: {chunk['title']}")
        print(f"{'─' * 60}")
        # Show first 300 chars of the chunk
        preview = chunk["text"][:300]
        print(preview)
        if len(chunk["text"]) > 300:
            print("...")
        print()

    print("✅ Query complete!")


if __name__ == "__main__":
    main()
