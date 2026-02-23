"""
Session 1: The Waterfall Pipeline — Demonstrating the Latency Problem
=====================================================================
This script shows the "old way" of building a voice pipeline:
  1. Record audio from microphone
  2. Send to ElevenLabs STT — wait for full transcription
  3. Send transcription to Gemini LLM — wait for full response
  4. Send response to ElevenLabs TTS — wait for full audio
  5. Play audio

Each step waits for the previous one to finish. This is SLOW on purpose.
That's the teaching point — we need streaming (Session 2) to fix this.

Usage:
  python session1_basics/04_basic_pipeline.py
"""

import os
import sys
import time
import io
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


# ============================================================
# Step 0: Setup
# ============================================================

def setup_clients():
    """Initialize API clients for ElevenLabs and Google Gemini."""
    from google import genai
    from elevenlabs import ElevenLabs

    gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

    return gemini_client, elevenlabs_client


# ============================================================
# Step 1: Record Audio
# ============================================================

def record_audio(duration: int = 5, sample_rate: int = 16000) -> bytes:
    """Record audio from the microphone."""
    import sounddevice as sd

    print(f"\n🎤 Recording for {duration} seconds... Speak now!")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("🎤 Recording complete!")

    # Convert to WAV bytes
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    buffer.seek(0)
    return buffer.read()


# ============================================================
# Step 2: Speech-to-Text (ElevenLabs)
# ============================================================

def speech_to_text(client, audio_bytes: bytes) -> str:
    """Send audio to ElevenLabs STT and wait for full transcription."""
    print("\n📝 Sending audio to ElevenLabs STT...")

    start = time.perf_counter()
    result = client.speech_to_text.convert(
        file=audio_bytes,
        model_id="scribe_v1",
        language_code="ar",
    )
    elapsed = time.perf_counter() - start

    text = result.text
    print(f"📝 STT Result: \"{text}\"")
    print(f"⏱️  STT took: {elapsed:.2f}s")
    return text, elapsed


# ============================================================
# Step 3: LLM Response (Google Gemini)
# ============================================================

def get_llm_response(client, text: str) -> str:
    """Send text to Gemini LLM and wait for full response."""
    print("\n🧠 Sending text to Gemini LLM...")

    start = time.perf_counter()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"You are a helpful voice assistant. Respond briefly in 2-3 sentences. The user said: {text}",
    )
    elapsed = time.perf_counter() - start

    reply = response.text
    print(f"🧠 LLM Response: \"{reply}\"")
    print(f"⏱️  LLM took: {elapsed:.2f}s")
    return reply, elapsed


# ============================================================
# Step 4: Text-to-Speech (ElevenLabs)
# ============================================================

def text_to_speech(client, text: str) -> bytes:
    """Send text to ElevenLabs TTS and wait for full audio."""
    print("\n🔊 Sending text to ElevenLabs TTS...")

    start = time.perf_counter()
    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # "George" — a default ElevenLabs voice
        model_id="eleven_flash_v2_5",
        output_format="pcm_16000",
    )
    # Collect all audio chunks
    audio_bytes = b"".join(audio_generator)
    elapsed = time.perf_counter() - start

    print(f"🔊 TTS generated {len(audio_bytes)} bytes of audio")
    print(f"⏱️  TTS took: {elapsed:.2f}s")
    return audio_bytes, elapsed


# ============================================================
# Step 5: Play Audio
# ============================================================

def play_audio(audio_bytes: bytes):
    """Play raw PCM audio bytes through speakers."""
    import sounddevice as sd

    print("\n🔊 Playing response audio...")
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    sd.play(audio_data, samplerate=16000)
    sd.wait()


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("🐢 THE WATERFALL PIPELINE — Demonstrating the Latency Problem")
    print("=" * 60)
    print("Each step waits for the previous one to finish completely.")
    print("Watch the timestamps — this is why we need streaming!\n")

    # Setup
    gemini_client, elevenlabs_client = setup_clients()

    # Check if user wants to type instead of record
    print("Options:")
    print("  [1] Record from microphone (5 seconds)")
    print("  [2] Type a message instead")
    choice = input("\nChoose (1 or 2): ").strip()

    pipeline_start = time.perf_counter()

    if choice == "1":
        # Step 1: Record
        audio_bytes = record_audio()
        # Step 2: STT
        transcript, stt_time = speech_to_text(elevenlabs_client, audio_bytes)
    else:
        # Skip recording — user types directly
        transcript = input("\n✍️  Type your message: ").strip()
        stt_time = 0.0
        print(f"(Skipping STT — using typed input)")

    if not transcript:
        print("No input received. Exiting.")
        return

    # Step 3: LLM
    response_text, llm_time = get_llm_response(gemini_client, transcript)

    # Step 4: TTS
    audio_output, tts_time = text_to_speech(elevenlabs_client, response_text)

    pipeline_total = time.perf_counter() - pipeline_start

    # Step 5: Play audio
    try:
        play_audio(audio_output)
    except Exception as e:
        print(f"⚠️  Could not play audio: {e}")
        print("   (Audio was generated successfully — playback requires sounddevice)")

    # Summary
    print("\n" + "=" * 60)
    print("📊 LATENCY BREAKDOWN")
    print("=" * 60)
    if stt_time > 0:
        print(f"  🎤 STT (ElevenLabs):     {stt_time:.2f}s")
    print(f"  🧠 LLM (Gemini):         {llm_time:.2f}s")
    print(f"  🔊 TTS (ElevenLabs):     {tts_time:.2f}s")
    print(f"  {'─' * 40}")
    print(f"  ⏱️  TOTAL PIPELINE:       {pipeline_total:.2f}s")
    print("=" * 60)
    print("\n😴 That's a LOT of waiting! In a real conversation,")
    print("   even 1 second of silence feels broken.")
    print("   In Session 2, we'll fix this with STREAMING.")
    print("   All steps will run in parallel — real-time response! 🚀\n")


if __name__ == "__main__":
    main()
