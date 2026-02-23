"""
RAG: Build the Annoy Vector Index
==================================
Reads the Arabic knowledge base, splits it into chunks,
embeds each chunk using Gemini text-embedding-004,
and builds an Annoy index for fast similarity search.
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types  # Added for embedding configuration
from annoy import AnnoyIndex

load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
N_TREES = 10
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.md"
INDEX_DIR = Path(__file__).parent / "index"


def read_knowledge_base(path: Path) -> str:
    """Read the knowledge base markdown file."""
    print(f"📄 Reading knowledge base: {path}")
    text = path.read_text(encoding="utf-8")
    print(f"   {len(text)} characters read")
    return text


def chunk_by_headers(text: str) -> list[dict]:
    """
    Split markdown text into chunks based on ## headers.
    """
    print("✂️  Splitting into chunks by headers...")

    # Split on ## headers
    sections = re.split(r'\n(?=##\s)', text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section or len(section) < 20:
            continue

        # Extract title from first line
        lines = section.split("\n")
        title = lines[0].strip("# ").strip()

        chunks.append({
            "title": title,
            "text": section,
            "word_count": len(section.split()),
        })

    print(f"   Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"   [{i}] {chunk['title'][:50]:50s} ({chunk['word_count']} words)")

    return chunks


def embed_chunks(chunks: list[dict], client: genai.Client) -> list[list[float]]:
    """Embed all chunks using Gemini gemini-embedding-001 with RETRIEVAL_DOCUMENT task type."""
    print(f"\n🔢 Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    embeddings = []

    for i, chunk in enumerate(chunks):
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk["text"],
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIM,
                task_type="RETRIEVAL_DOCUMENT",
                title=chunk["title"],
            ),
        )
        embeddings.append(response.embeddings[0].values)
        print(f"   [{i}] Embedded: {chunk['title'][:50]}")

    print(f"   Done! {len(embeddings)} embeddings (dim={EMBEDDING_DIM})")
    return embeddings


def build_annoy_index(embeddings: list[list[float]]) -> AnnoyIndex:
    """Build an Annoy index from embeddings."""
    print(f"\n🏗️  Building Annoy index ({N_TREES} trees)...")

    index = AnnoyIndex(EMBEDDING_DIM, "angular")

    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)

    index.build(N_TREES)
    print(f"   Index built with {len(embeddings)} vectors")
    return index


def save_index(index: AnnoyIndex, chunks: list[dict]):
    """Save the Annoy index and chunks to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "annoy_index.ann"
    chunks_path = INDEX_DIR / "chunks.json"

    index.save(str(index_path))
    print(f"\n💾 Index saved: {index_path}")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 Chunks saved: {chunks_path}")


def main():
    print("=" * 60)
    print("🧠 RAG INDEX BUILDER — EEExplore Knowledge Base")
    print("=" * 60)

    # Initialize Gemini client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # Step 1: Read knowledge base
    text = read_knowledge_base(KNOWLEDGE_BASE_PATH)

    # Step 2: Chunk by headers
    chunks = chunk_by_headers(text)

    # Step 3: Embed chunks
    embeddings = embed_chunks(chunks, client)

    # Step 4: Build Annoy index
    index = build_annoy_index(embeddings)

    # Step 5: Save everything
    save_index(index, chunks)

    print(f"\n{'=' * 60}")
    print(f"✅ RAG INDEX BUILT SUCCESSFULLY")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Dimension: {EMBEDDING_DIM}")
    print(f"   Trees: {N_TREES}")
    print(f"{'=' * 60}")
    print(f"\nTest it: python rag/query_index.py \"شنو المسارات المتاحة في EEExtra؟\"")


if __name__ == "__main__":
    main()