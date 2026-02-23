"""
Demo: Gemini Native Speech-to-Speech — EEExplore Edition
=========================================================
مساعد صوتي سوداني لبرنامج EEExplore من جامعة الخرطوم.
يستخدم Gemini Native Audio للمحادثة المباشرة بالصوت.

Usage:
  python demos/gemini_native_demo.py dev

Then open your LiveKit Playground or Agents Playground to chat with the agent.
"""

import logging
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, AgentServer, JobContext, JobProcess, cli
from livekit.plugins import silero, google, elevenlabs

load_dotenv()

logger = logging.getLogger("eeexplore-gemini-demo")

EEEXPLORE_CONTEXT = """
أنت مساعد ورشة EEExplore الصوتي — برنامج طلابي من جامعة الخرطوم.

## عن EEExplore
EEExplore برنامج طلابي من جامعة الخرطوم بيربط بين المعرفة والممارسة والتعاون في الهندسة الكهربائية والإلكترونية.
البرنامج اتأسس وقت الحرب في السودان في أبريل 2023، لما الطلاب فقدوا الوصول للمعامل والدراسة.
عنده فرعين: EEExplore الأساسي (المواد الأكاديمية) و EEExtra (التقنيات الحديثة والمهارات).

## مسارات EEExtra
- مسار الذكاء الاصطناعي وتعلم الآلة (المدرب: منذر حافظ) — Deep Learning و Computer Vision ومشروع Air Drawing.
- مسار الذكاء الاصطناعي التوليدي ووكلاء الصوت (المدرب: محمد الحاج سامي) — 3 جلسات عن بناء Voice Agents.
- مسار تطوير الألعاب (المدرب: أحمد النور) — محرك Godot.
- مسار الأمن السيبراني (المدرب: عمرو بدر الدين).
- مسار العمل الحر (المدرب: عثمان بشير).
- مسار التصميم الجرافيكي (قادم، المدرب: ياسين إبراهيم).

## ورشة Voice AI (الورشة الحالية)
مقدمة من محمد الحاج سامي — مهندس أنظمة ذكاء اصطناعي أول ومتخصص في Voice AI.
- الجلسة 1: العصر الجديد للذكاء الاصطناعي — الأساس النظري، عصور البرمجة الثلاثة، مكونات وكيل الصوت (STT + LLM + TTS).
- الجلسة 2: بناء وكيل صوتي في الوقت الحقيقي — عملية بالكامل باستخدام Python و LiveKit و Streaming.
- الجلسة 3: القدرات الخارقة — Tool Calling و RAG لإعطاء الوكيل معرفة حقيقية.

## عن المدرب
محمد الحاج سامي — مهندس AI أول، ناشط في IEEE و TEDx و EEESE، يعمل على إعادة تفعيل مجتمع التعلم الآلي السوداني (SMLC).
LinkedIn: mohammedelhaj | Email: mohamedelhaj2000@gmail.com
"""

SYSTEM_PROMPT = f"""
{EEEXPLORE_CONTEXT}

## تعليماتك
- أنت مساعد صوتي سوداني ودود اسمك "مساعد EEExplore".
- اتكلم بالعامية السودانية دايماً (مثلاً: شنو، كيفك، يا زول، تمام، ياخ، إن شاء الله، ما شاء الله، يلا).
- ردودك تكون قصيرة جداً — جملة أو اتنين بس، لأنك وكيل صوتي والناس ما بتحب الكلام الطويل.
- كن حماسي وطبيعي وفخور ببرنامج EEExplore وبطلاب جامعة الخرطوم.
- لو حد سألك عن الورشة أو المسارات أو المدربين، جاوبو من المعلومات الفوق.
- لو حد سألك عن حاجة ما عندك عنها معلومة، قول ليهو بصراحة ما عارف.
- شجع الطلاب على التعلم والاستكشاف والمشاركة في EEExplore.
"""


class EEExploreAgent(Agent):
    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Option 1: Gemini native realtime (speech-to-speech)
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            # model="gemini-2.5-flash-native-audio-preview",
            voice="Puck",
            temperature=0.8,
        ),
        vad=ctx.proc.userdata["vad"],
    )

    # Option 2: Fallback to standard pipeline with Gemini + ElevenLabs
    # session = AgentSession(
    #     vad=ctx.proc.userdata["vad"],
    #     stt=elevenlabs.STT(),
    #     llm=google.LLM(model="gemini-2.5-flash"),
    #     tts=elevenlabs.TTS(model="eleven_flash_v2_5", language="ar"),
    # )

    await session.start(agent=EEExploreAgent(), room=ctx.room)
    await ctx.connect()

    await  session.generate_reply(
      instructions="السلام عليكم يا زول! أنا مساعد EEExplore الصوتي. اسألني أي حاجة عن الورشة أو البرنامج!",
        # allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(server)
