# Session 2: From Text to Talk — Building a Real-Time Agent

## What You'll Learn
- Real-time voice architecture (streaming vs waterfall)
- The 4 components: VAD, STT, LLM, TTS
- Building a voice agent with ~20 lines of Python
- System prompt design for voice AI

## Files

### `01_basic_agent.py`
The **minimal voice agent** — this is what we build live on camera in Session 2. About 20 lines of actual code using LiveKit Agents SDK.

Components: Silero VAD, ElevenLabs STT, Google Gemini LLM, ElevenLabs TTS.

```bash
python session2_realtime_agent/01_basic_agent.py dev
```

### `02_agent_with_personality.py`
Same agent with the **EEExplore Arabic personality**. Demonstrates how the system prompt transforms the agent's behavior.

```bash
python session2_realtime_agent/02_agent_with_personality.py dev
```

## Prerequisites
- Python 3.11+
- API keys in `.env`: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `GOOGLE_API_KEY`, `ELEVENLABS_API_KEY`
- Run `pip install -r requirements.txt`

## Testing
1. Run the agent with `dev` flag
2. Open [LiveKit Playground](https://agents-playground.livekit.io/) in your browser
3. Connect to your LiveKit project
4. Start talking!
