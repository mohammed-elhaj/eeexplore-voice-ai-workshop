# Session 3: Giving Your Agent Superpowers

## What You'll Learn
- Generative AI vs Agentic AI
- The ReAct reasoning loop (Reason → Act → Observe)
- Tool calling — giving your agent real-world capabilities
- RAG (Retrieval Augmented Generation) — custom knowledge

## Files

### `01_agent_with_search.py`
Agent + **web search tool** using DuckDuckGo. This is what we build live in Session 3.

```bash
python session3_superpowers/01_agent_with_search.py dev
```

### `02_agent_with_rag.py`
Agent + **RAG** using Annoy vector database. Searches the EEExplore knowledge base.

> **Note:** Run `python rag/build_index.py` first to build the vector index!

```bash
python session3_superpowers/02_agent_with_rag.py dev
```

### `03_agent_full.py`
The **final agent** with all 3 superpowers: RAG + Web Search + Feedback collection.

```bash
python session3_superpowers/03_agent_full.py dev
```

### `tools/`
Reusable tool modules:
- `web_search.py` — DuckDuckGo search utility
- `feedback.py` — Feedback collection (saves to JSON)

## Prerequisites
- All API keys in `.env`
- For RAG: run `python rag/build_index.py` first
- `pip install -r requirements.txt`
