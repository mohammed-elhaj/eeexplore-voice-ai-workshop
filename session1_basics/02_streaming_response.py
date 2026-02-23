"""
Session 1: Streaming — Watch the AI Think Token by Token
==========================================================
LLMs don't write the full answer and then send it.
They generate ONE TOKEN (word/piece) at a time, like typing.

This script shows you that process in real-time:
  - Without streaming: you wait... wait... then get everything at once
  - With streaming: you see each word appear as it's generated

This is the KEY CONCEPT behind real-time voice agents!
  Instead of waiting for the full response, we start speaking
  as soon as the first few words are ready.

Usage:
  python session1_basics/02_streaming_response.py
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent.parent / ".env")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

QUESTION = "Explain how a car engine works in simple terms."


# ============================================================
# Part 1: Without Streaming (the slow way)
# ============================================================

print("=" * 50)
print("Part 1: WITHOUT streaming")
print("=" * 50)
print(f"\nQuestion: {QUESTION}")
print("Waiting for full response...\n")

start = time.perf_counter()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=QUESTION,
)

elapsed = time.perf_counter() - start

print(f"Response: {response.text}")
print(f"\nTime to first word: {elapsed:.2f}s (you waited this long!)")


# ============================================================
# Part 2: With Streaming (the fast way)
# ============================================================

print("\n" + "=" * 50)
print("Part 2: WITH streaming")
print("=" * 50)
print(f"\nQuestion: {QUESTION}")
print("Watch the words appear one by one...\n")

start = time.perf_counter()
first_token_time = None
token_count = 0

# generate_content_stream returns chunks as they're generated
response_stream = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=QUESTION,
)

print("Response: ", end="", flush=True)

for chunk in response_stream:
    if chunk.text:
        # Record when the first token arrives
        if first_token_time is None:
            first_token_time = time.perf_counter() - start

        print(chunk.text, end="", flush=True)
        token_count += 1

total_time = time.perf_counter() - start

print(f"\n\nTime to FIRST word:  {first_token_time:.2f}s")
print(f"Time to FULL answer: {total_time:.2f}s")
print(f"Chunks received:     {token_count}")


# ============================================================
# Part 3: Why This Matters for Voice AI
# ============================================================

print("\n" + "=" * 50)
print("Why streaming matters for voice agents")
print("=" * 50)

print("""
Without streaming (Session 1 pipeline):
  User speaks → [wait for STT] → [wait for LLM] → [wait for TTS] → hear response
  Total wait: 3-5 seconds of silence!

With streaming (Session 2 real-time agent):
  User speaks → STT streams text → LLM streams tokens → TTS speaks immediately
  Total wait: ~0.5 seconds!

The trick: we don't wait for the full answer.
As soon as the first sentence is ready, we start converting it to speech.
This is why real-time voice agents feel like talking to a real person.
""")


# ============================================================
# Part 4: Interactive streaming chat
# ============================================================

print("=" * 50)
print("Try it yourself! Ask anything and watch it stream.")
print("Type 'quit' to exit.")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break
    if not user_input:
        continue

    start = time.perf_counter()
    first_token_time = None

    stream = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=user_input,
    )

    print("Gemini: ", end="", flush=True)
    for chunk in stream:
        if chunk.text:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            print(chunk.text, end="", flush=True)

    total = time.perf_counter() - start
    if first_token_time:
        print(f"\n  (first token: {first_token_time:.2f}s | total: {total:.2f}s)")
