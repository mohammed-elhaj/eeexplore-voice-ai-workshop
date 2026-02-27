"""
Session 3: The Full Agent — RAG + Web Search + Feedback
========================================================
The final, complete voice agent with all 3 superpowers:
  1. RAG tool — searches the EEExplore knowledge base (Annoy VDB)
  2. Web search tool — searches the live internet (Tavily)
  3. Feedback tool — collects user feedback (saves to JSON)

IMPORTANT: Run `python rag/build_index.py` first to build the RAG index!
Requires TAVILY_API_KEY in your .env file.

Usage:
  python session3_superpowers/03_agent_full.py dev
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

# Add parent dir to path so we can import from tools/
sys.path.insert(0, str(Path(__file__).parent))
from tools.web_search import search_tavily
from tools.feedback import save_feedback

load_dotenv()

logger = logging.getLogger("voice-agent")

# ============================================================
# RAG Setup
# ============================================================

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
RAG_INDEX_DIR = Path(__file__).parent.parent / "rag" / "index"


def load_rag_index():
    """Load the pre-built Annoy index and chunks."""
    index_path = RAG_INDEX_DIR / "annoy_index.ann"
    chunks_path = RAG_INDEX_DIR / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        logger.warning("⚠️ RAG index not found! Run: python rag/build_index.py")
        return None, None

    index = AnnoyIndex(EMBEDDING_DIM, "angular")
    index.load(str(index_path))

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"🧠 RAG index loaded: {len(chunks)} chunks")
    return index, chunks


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


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """أنت "أحمد"، المساعد الصوتي الذكي لورشة EEExplore لبناء وكلاء الصوت بالذكاء الاصطناعي.
شخصيتك: شاب سوداني متحمس للتكنولوجيا، بتتكلم بالعربي السوداني بأسلوب ودود وحماسي. بتحب تشجع الناس وتحفزهم يتعلموا.

قدراتك:
- عندك معرفة عن ورشة بناء وكلاء الصوت وعن برنامج EEExtra بكل مساراته. استخدم أداة البحث في قاعدة المعرفة لما حد يسألك عن الورشة أو البرنامج.
- بتقدر تبحث في الإنترنت عن معلومات حية وحديثة باستخدام أداة البحث في الإنترنت.
- بتقدر تجمع ملاحظات من المستخدمين عن الورشة. لما حد يعبر عن رأيه أو يبغى يدي ملاحظات، اسأله عن اسمه وملاحظاته واستخدم أداة جمع التغذية الراجعة.
-عندما تستعمل اداة البحث في الانترنت search_web و احصرص على ان تكون ال query باللغة الانجليزية عشان النتائج تطلع كويسة. 

قواعد مهمة:
- ردودك قصيرة ومختصرة — جملتين أو ثلاثة بالكتير.
- لو ما بتعرف الإجابة، قول بصراحة.
- اكتب الأرقام كلها بالحروف العربية عشان النطق يكون واضح. مثلاً: اكتب "ثلاثة" بدل "3"، و"ألفين وخمسة وعشرين" بدل "2025"، و"خمسة وأربعين بالمية" بدل "45%".
- ما تستخدم قوائم أو نقاط أو تنسيق — ده وكيل صوتي، الناس بتسمع مش بتقرأ.
- ما تستخدم رموز أو إيموجي في ردودك."""


# ============================================================
# Agent with All Tools
# ============================================================


class FullAgent(Agent):
    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)

    # Tool 1: Knowledge Base Search (RAG)
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

        query_embedding = embed_query(query)
        indices, distances = rag_index.get_nns_by_vector(
            query_embedding, 3, include_distances=True
        )

        results = []
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            chunk_text = rag_chunks[idx]["text"]
            results.append(f"[Result {i+1}]\n{chunk_text}")

        context = "\n\n".join(results)
        print(f"🧠 RAG found {len(results)} relevant chunks")
        return f"Here is the relevant information from the knowledge base:\n\n{context}"

    # Tool 2: Web Search
    @function_tool
    async def search_web(self, ctx: RunContext, query: str):
        """Search the web for current, up-to-date information using Tavily.
        Use this when the user asks about recent events, news, or anything
        not related to the workshop that you don't know from your training data.

        Args:
            query: The search query to look up online.
        """
        results = search_tavily(query)
        return results

    # Tool 3: Feedback Collection
    @function_tool
    async def collect_feedback(self, ctx: RunContext, name: str, feedback: str):
        """Collect feedback from the user about the workshop. Call this when
        the user wants to give feedback or share their opinion. You need
        their name and their feedback comment.

        Args:
            name: The user's name.
            feedback: The user's feedback comment about the workshop.
        """
        result = save_feedback(name, feedback)
        return result


# ============================================================
# Server Setup
# ============================================================

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
                                               voice_id="yXEnnEln9armDCyhkXcA",
),
    )

    await session.start(agent=FullAgent(), room=ctx.room)
    await ctx.connect()

    await session.say(
        "يا هلا وسهلا! أنا أحمد، مساعدك الذكي من EEExplore. بقدر أجاوبك عن الورشة، "
        "أبحث ليك في الإنترنت، وأجمع ملاحظاتك. تفضل اسألني أي حاجة!",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(server)
