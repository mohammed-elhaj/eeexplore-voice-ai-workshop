"""
Session 3: Voice Agent + RAG (Retrieval Augmented Generation)
=============================================================
Agent with a knowledge base about the EEExplore workshop.
Uses Annoy vector database for fast similarity search.

RAG is implemented as a TOOL — the LLM decides when to search
the knowledge base, just like it decides when to search the web.

IMPORTANT: Run `python rag/build_index.py` first to build the index!

Usage:
  python session3_superpowers/02_agent_with_rag.py dev
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from annoy import AnnoyIndex
from google import genai as genai_client
from google.genai import types
from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    JobProcess,
    RunContext,
    function_tool,
    cli,
)
from livekit.plugins import silero, elevenlabs, google

load_dotenv()

logger = logging.getLogger("voice-agent")

# RAG Configuration
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
RAG_INDEX_DIR = Path(__file__).parent.parent / "rag" / "index"

SYSTEM_PROMPT = """أنت "أحمد"، المساعد الصوتي الذكي لورشة EEExplore لبناء وكلاء الصوت بالذكاء الاصطناعي.
شخصيتك: شاب سوداني متحمس للتكنولوجيا، بتتكلم بالعربي السوداني بأسلوب ودود وحماسي. بتحب تشجع الناس وتحفزهم يتعلموا.

عندك معرفة عن ورشة بناء وكلاء الصوت وعن برنامج EEExtra بكل مساراته.
استخدم أداة البحث في قاعدة المعرفة لما حد يسألك عن الورشة أو البرنامج.

قواعد مهمة:
- ردودك قصيرة ومختصرة — جملتين أو ثلاثة بالكتير.
- لو ما بتعرف الإجابة، قول بصراحة.
- اكتب الأرقام كلها بالحروف العربية عشان النطق يكون واضح. مثلاً: اكتب "ثلاثة" بدل "3"، و"ألفين وخمسة وعشرين" بدل "2025"، و"خمسة وأربعين بالمية" بدل "45%".
- ما تستخدم قوائم أو نقاط أو تنسيق — ده وكيل صوتي، الناس بتسمع مش بتقرأ.
- ما تستخدم رموز أو إيموجي في ردودك."""


def load_rag_index():
    """Load the pre-built Annoy index and chunks."""
    index_path = RAG_INDEX_DIR / "annoy_index.ann"
    chunks_path = RAG_INDEX_DIR / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        logger.error("❌ RAG index not found! Run: python rag/build_index.py")
        return None, None

    # Load Annoy index
    index = AnnoyIndex(EMBEDDING_DIM, "angular")
    index.load(str(index_path))

    # Load text chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"🧠 RAG index loaded: {len(chunks)} chunks")
    return index, chunks


# Load RAG index at module level
rag_index, rag_chunks = load_rag_index()
gemini_embed_client = genai_client.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


def embed_query(query: str) -> list[float]:
    """Embed a query using Gemini gemini-embedding-001 with RETRIEVAL_QUERY task type."""
    response = gemini_embed_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return response.embeddings[0].values


class RAGAgent(Agent):
    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool
    async def search_knowledge_base(self, ctx: RunContext, query: str):
        """Search the EEExplore workshop knowledge base for information about
        the sessions, topics, tracks, and instructors. Use this when the user
        asks about the workshop content.

        Args:
            query: The question or topic to search for in the knowledge base.
        """
        if rag_index is None or rag_chunks is None:
            return "Knowledge base is not available. Please run build_index.py first."

        print(f"🧠 RAG search: {query}")

        # Embed the query
        query_embedding = embed_query(query)

        # Search Annoy index for top 3 nearest chunks
        indices, distances = rag_index.get_nns_by_vector(
            query_embedding, 3, include_distances=True
        )

        # Retrieve the matching chunks
        results = []
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            chunk_text = rag_chunks[idx]["text"]
            results.append(f"[Result {i+1}] (distance: {dist:.3f})\n{chunk_text}")

        context = "\n\n".join(results)
        print(f"🧠 RAG found {len(results)} relevant chunks")
        return f"Here is the relevant information from the knowledge base:\n\n{context}"


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=elevenlabs.STT(),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=elevenlabs.TTS(model="eleven_flash_v2_5", language="ar",
                           voice_id="yXEnnEln9armDCyhkXcA",),
    )

    await session.start(agent=RAGAgent(), room=ctx.room)
    await ctx.connect()

    await session.say(
        "يا هلا! أنا أحمد، مساعدك الصوتي من EEExplore. بقدر أجاوبك عن الورشة والمسارات والمواضيع. اسألني أي حاجة!",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(server)
