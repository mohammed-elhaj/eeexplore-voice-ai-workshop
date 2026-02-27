"""
Session 2: Voice Agent with EEExplore Arabic Personality
========================================================
Same agent as 01_basic_agent.py but with a polished Arabic
system prompt. Demonstrates how the system prompt transforms
the agent's personality and behavior.

The agent speaks in Sudanese Arabic dialect, keeps responses
short and conversational — optimized for voice interaction.

Usage:
  python session2_realtime_agent/02_agent_with_personality.py dev
"""

import logging
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, AgentServer, JobContext, JobProcess, cli
from livekit.plugins import silero, elevenlabs, google

load_dotenv()

logger = logging.getLogger("voice-agent")

# Arabic system prompt — the agent's personality
SYSTEM_PROMPT = """أنت "أحمد"، المساعد الصوتي الذكي لورشة EEExplore لبناء وكلاء الصوت بالذكاء الاصطناعي.
شخصيتك: شاب سوداني متحمس للتكنولوجيا، بتتكلم بالعربي السوداني بأسلوب ودود وحماسي. بتحب تشجع الناس وتحفزهم يتعلموا.

قواعد مهمة:
- ردودك قصيرة ومختصرة — جملتين أو ثلاثة بالكتير.
- لو سألوك عن حاجة ما بتعرفها، قول بصراحة إنك ما بتعرف.
- اكتب الأرقام كلها بالحروف العربية عشان النطق يكون واضح. مثلاً: اكتب "ثلاثة" بدل "3"، و"ألفين وخمسة وعشرين" بدل "2025"، و"خمسة وأربعين بالمية" بدل "45%".
- ما تستخدم قوائم أو نقاط أو تنسيق — ده وكيل صوتي، الناس بتسمع مش بتقرأ.
- ما تستخدم رموز أو إيموجي في ردودك."""


class EEExploreAgent(Agent):
    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="eeexplore-personality")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=elevenlabs.STT(),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=elevenlabs.TTS(model="eleven_flash_v2_5", language="ar",
                     voice_id="yXEnnEln9armDCyhkXcA"      
                           ),
    )

    await session.start(agent=EEExploreAgent(), room=ctx.room)
    await ctx.connect()

    # Arabic greeting
    await session.say(
        "يا هلا وسهلا! أنا أحمد، مساعدك الصوتي من ورشة EEExplore. جاهز أساعدك وأجاوب على أي سؤال عندك. تفضل!",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(server)
