"""
Session 3: Voice Agent + Web Search Tool
=========================================
This is what we build live in Session 3. Starting from the Session 2
agent, we add a web search tool using Tavily.

The agent can now search the live internet to answer questions about
current events, recent news, or anything not in its training data.

Requires TAVILY_API_KEY in your .env file.

Usage:
  python session3_superpowers/01_agent_with_search.py dev
"""

import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
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

load_dotenv()

logger = logging.getLogger("voice-agent")

SYSTEM_PROMPT = """أنت "أحمد"، المساعد الصوتي الذكي لورشة EEExplore لبناء وكلاء الصوت بالذكاء الاصطناعي.
شخصيتك: شاب سوداني متحمس للتكنولوجيا، بتتكلم بالعربي السوداني بأسلوب ودود وحماسي. بتحب تشجع الناس وتحفزهم يتعلموا.

قواعد مهمة:
- ردودك قصيرة ومختصرة — جملتين أو ثلاثة بالكتير.
- search_web لو سألوك عن حاجة ما بتعرفها أو محتاجة معلومات حديثة، استخدم أداة البحث في الإنترنت.
عندما تستعمل اداة البحث في الانترنت search_web و احصرص على ان تكون ال query باللغة الانجليوية عشان النتائج تطلع كويسة. 
- لو ما بتعرف الإجابة حتى بعد البحث، قول بصراحة.
- اكتب الأرقام كلها بالحروف العربية عشان النطق يكون واضح. مثلاً: اكتب "ثلاثة" بدل "3"، و"ألفين وخمسة وعشرين" بدل "2025"، و"خمسة وأربعين بالمية" بدل "45%".
- ما تستخدم قوائم أو نقاط أو تنسيق — ده وكيل صوتي، الناس بتسمع مش بتقرأ.
- ما تستخدم رموز أو إيموجي في ردودك."""


class SearchAgent(Agent):
    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool
    async def search_web(self, ctx: RunContext, query: str):
        """Search the web for current, up-to-date information using Tavily.
        Use this when the user asks about recent events, news, or anything
        you don't have in your training data.

        Args:
            query: The search query to look up online.
        """
        results = search_tavily(query)
        return results
    



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
        tts=elevenlabs.TTS(model="eleven_flash_v2_5", language="ar", voice_id="yXEnnEln9armDCyhkXcA"),
    )

    await session.start(agent=SearchAgent(), room=ctx.room)
    await ctx.connect()

    await session.say(
        "يا هلا! أنا أحمد، مساعدك الصوتي من EEExplore. بقدر أبحث ليك في الإنترنت عن أي معلومة حديثة. اسألني!",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(server)
