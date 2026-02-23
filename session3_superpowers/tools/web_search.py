"""
Web Search Tool — Tavily
=========================
Searches the web using Tavily AI search API.
Used by the voice agent to access live internet information.

Requires TAVILY_API_KEY in your .env file.
"""

import os
from tavily import TavilyClient


def search_tavily(query: str) -> str:
    """
    Search the web using Tavily and return the answer with sources.

    Args:
        query: The search query to look up.

    Returns:
        Formatted string with the search answer and sources.
    """
    print(f"🔍 Searching web for: {query}")

    client = TavilyClient(os.environ["TAVILY_API_KEY"])
    response = client.search(
        query=query,
        include_answer="basic",
        search_depth="advanced",
    )

    # Extract the answer and top results
    answer = response.get("answer", "")
    results = response.get("results", [])

    if answer:
        formatted = f"Answer: {answer}\n\nSources:\n"
        formatted += "\n".join(
            [f"- {r['title']}: {r['url']}" for r in results[:3]]
        )
    elif results:
        formatted = "\n".join(
            [f"- {r['title']}: {r.get('content', '')[:200]}" for r in results[:3]]
        )
    else:
        formatted = "No results found for this query."

    print(f"🔍 Search complete")
    return formatted
