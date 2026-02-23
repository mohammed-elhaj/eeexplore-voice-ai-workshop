"""
Session 1: Chat with Gemini — Multi-Turn Conversations
========================================================
In the previous example, each question was independent.
But real conversations have CONTEXT — the AI should remember
what you said earlier.

This script shows how to build a conversation with history.
Think of it like a WhatsApp chat — each message builds on the last.

Key concept:
  - LLMs don't actually "remember" anything
  - WE send the entire conversation history every time
  - The AI reads all previous messages to understand context

Usage:
  python session1_basics/01_chat_with_gemini.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent.parent / ".env")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ============================================================
# Part 1: The Problem — No Memory
# ============================================================

print("=" * 50)
print("Part 1: Without conversation history (no memory)")
print("=" * 50)

# First question
response1 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="My name is Ahmed and I study at University of Khartoum.",
)
print(f"\nYou:    My name is Ahmed and I study at University of Khartoum.")
print(f"Gemini: {response1.text}")

# Second question — Gemini has NO IDEA who we are!
response2 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is my name and where do I study?",
)
print(f"\nYou:    What is my name and where do I study?")
print(f"Gemini: {response2.text}")
print("\n>> Gemini doesn't remember! Each call is independent.")


# ============================================================
# Part 2: The Solution — Send Chat History
# ============================================================

print("\n" + "=" * 50)
print("Part 2: With conversation history (has memory!)")
print("=" * 50)

# We build the conversation as a list of messages.
# Each message has a "role" (user or model) and "parts" (the text).
# We send ALL messages every time so Gemini can see the full context.

conversation = [
    {"role": "user", "parts": [{"text": "My name is Ahmed and I study at University of Khartoum."}]},
    {"role": "model", "parts": [{"text": "Nice to meet you Ahmed! That's great that you're studying at the University of Khartoum."}]},
    {"role": "user", "parts": [{"text": "What is my name and where do I study?"}]},
]

response3 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=conversation,
)
print(f"\nYou:    What is my name and where do I study?")
print(f"Gemini: {response3.text}")
print("\n>> Now Gemini remembers! Because we sent the full conversation.")


# ============================================================
# Part 3: System Instructions — Give the AI a personality
# ============================================================

print("\n" + "=" * 50)
print("Part 3: System instructions (AI personality)")
print("=" * 50)

# The "system instruction" tells Gemini WHO it should be.
# It's like giving an actor a character description before they perform.

response4 = client.models.generate_content(
    model="gemini-2.5-flash",
    config={
        "system_instruction": (
            "You are a Sudanese AI assistant named Zool. "
            "You speak in a friendly, casual way. "
            "You love technology and always encourage students to learn. "
            "Keep responses to 2-3 sentences."
        ),
    },
    contents="Tell me about yourself.",
)
print(f"\nYou:  Tell me about yourself.")
print(f"Zool: {response4.text}")


# ============================================================
# Part 4: Interactive Chat — Full conversation with memory
# ============================================================

print("\n" + "=" * 50)
print("Part 4: Interactive chat (with memory!)")
print("Type 'quit' to exit.")
print("=" * 50)

# Start with empty history
chat_history = []

system_prompt = (
    "You are a helpful AI tutor teaching a workshop about AI and voice agents. "
    "Keep your responses short (2-3 sentences). Be encouraging and friendly."
)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if not user_input:
        continue

    # Add user message to history
    chat_history.append({"role": "user", "parts": [{"text": user_input}]})

    # Send full history to Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config={"system_instruction": system_prompt},
        contents=chat_history,
    )

    assistant_reply = response.text

    # Add Gemini's response to history (so it remembers next time)
    chat_history.append({"role": "model", "parts": [{"text": assistant_reply}]})

    print(f"Gemini: {assistant_reply}")

    # Show how many messages are in memory
    print(f"  (conversation history: {len(chat_history)} messages)")
