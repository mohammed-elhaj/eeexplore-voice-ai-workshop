"""
Restaurant Voice Agent
======================
A simple LiveKit voice agent that knows about restaurants.

Components:
  - VAD:  Silero (detects when user speaks)
  - STT:  Google Gemini (converts speech to text)
  - LLM:  Google Gemini (generates response)
  - TTS:  ElevenLabs (converts text to speech)

Usage:
  python restaurant_agent.py dev
"""

import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, cli
from livekit.plugins import silero, elevenlabs, google

load_dotenv()

logger = logging.getLogger("restaurant-agent")


class RestaurantAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a friendly restaurant recommendation assistant. "
                "You help people find places to eat, suggest dishes, and answer questions about cuisines. "
                "Keep your responses short — two or three sentences maximum. "
                "Speak in a warm, conversational tone like a foodie friend. "
                "Do not use lists, bullet points, or any formatting — this is a voice agent, "
                "people are listening, not reading. "
                "If someone asks about a specific cuisine, give a brief recommendation. "
                "If they ask what to order, suggest one or two dishes."
            ),
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=elevenlabs.STT(),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=elevenlabs.TTS(model="eleven_flash_v2_5"),
    )

    await session.start(agent=RestaurantAgent(), room=ctx.room)

    await session.say("Hey there! I'm your restaurant buddy. Craving anything specific, or want me to suggest something?")


if __name__ == "__main__":
    cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="restaurant-agent",
            prewarm_fnc=prewarm,
        )
    )
