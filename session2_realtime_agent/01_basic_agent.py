"""
Session 2: The Basic Voice Agent — ~20 Lines of Code
=====================================================
This is THE most important file in the workshop. We build this
live on camera to show how easy it is to create a real-time
voice agent using LiveKit Agents SDK.

Components:
  - VAD:  Silero (detects when user speaks)
  - STT:  ElevenLabs (converts speech to text)
  - LLM:  Google Gemini (generates response)
  - TTS:  ElevenLabs (converts text to speech)

Usage:
  python session2_realtime_agent/01_basic_agent.py dev
"""

import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, cli
from livekit.plugins import silero, elevenlabs, google

load_dotenv()

logger = logging.getLogger("voice-agent")


# Define the agent with a simple personality
class VoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. "
                "Keep your responses short and concise — two or three sentences maximum. "
                "Speak in a friendly, conversational tone. "
                "Do not use lists, bullet points, or any formatting — this is a voice agent, "
                "people are listening, not reading."
            ),
        )


# Prewarm: load VAD model once at startup
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# Handle each voice session
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Assemble the 4 components
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=elevenlabs.STT(model_id="scribe_v2_realtime",),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=elevenlabs.TTS(
             model="eleven_flash_v2_5",
                        #    voice_id="2ajXGJNYBR0iNHpS4VZb"
                           ),
        tools=[google.tools.GoogleSearch()]
    )

    # Start the agent
    await session.start(agent=VoiceAgent(), room=ctx.room)

    # Greet the user
    await session.say("Hello! I'm your AI assistant. How can I help you today?")


if __name__ == "__main__":
    cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # agent_name="eeexplore-basic",
            prewarm_fnc=prewarm,
        )
    )
