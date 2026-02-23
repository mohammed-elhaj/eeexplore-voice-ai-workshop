# Session 1: The New Age of AI

## What You'll Learn
- How to use the Gemini API (your first API call!)
- How LLMs handle conversations (chat history)
- How streaming works (token-by-token generation)
- Prompt engineering basics
- The voice agent pipeline: STT + LLM + TTS
- Why the simple waterfall approach is too slow

## Files (in learning order)

### `00_hello_gemini.py`
Your **first API call**. Send a question to Gemini, get an answer back. The simplest possible example — understand how the API works before anything else.

```bash
python session1_basics/00_hello_gemini.py
```

### `01_chat_with_gemini.py`
**Multi-turn conversations**. Learn how LLMs "remember" context — spoiler: they don't! We send the full chat history every time. Also covers system instructions (giving the AI a personality).

```bash
python session1_basics/01_chat_with_gemini.py
```

### `02_streaming_response.py`
**Streaming responses** — watch the AI generate text token by token in real-time. This is THE key concept behind real-time voice agents: don't wait for the full answer, start speaking as soon as the first words are ready.

```bash
python session1_basics/02_streaming_response.py
```

### `03_prompt_engineering.py`
**Prompt engineering** — the "Manager's Skill". See how lazy prompts vs engineered prompts produce dramatically different results. Includes a voice-specific prompting demo (chat prompts sound terrible when spoken!).

```bash
python session1_basics/03_prompt_engineering.py
```

### `04_basic_pipeline.py`
The **waterfall pipeline** — puts it all together with STT + LLM + TTS. This intentionally feels SLOW because each step waits for the previous one. That's the teaching point: we need streaming (Session 2) to fix this.

```bash
python session1_basics/04_basic_pipeline.py
```

## Prerequisites
- Python 3.11+
- API keys: `GOOGLE_API_KEY`, `ELEVENLABS_API_KEY` in `.env`
- For files 00-03, you only need `GOOGLE_API_KEY`
- File 04 also requires `ELEVENLABS_API_KEY` (for STT and TTS)
