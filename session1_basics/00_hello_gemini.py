"""
Session 1: Hello Gemini — Your First API Call
===============================================
This is the simplest possible example of using the Gemini API.
We send ONE question, and get ONE answer back. That's it.

Think of it like sending a text message to a very smart friend.

Usage:
  python session1_basics/00_hello_gemini.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv(Path(__file__).parent.parent / ".env")


# ============================================================
# Step 1: Create a client (your connection to Gemini)
# ============================================================
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# That's it! You now have a connection to Google's Gemini AI.
# The API key is like your password — it tells Google who you are.


# ============================================================
# Step 2: Send a message and get a response
# ============================================================

print("=" * 50)
print("Hello Gemini — Your First API Call")
print("=" * 50)

# This is the simplest API call possible:
# - model: which AI model to use (gemini-2.5-flash is fast and free)
# - contents: your message (just a plain string)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is artificial intelligence? Explain in 2 sentences.",
)

# The response object contains the AI's answer
print(f"\nQuestion: What is artificial intelligence?")
print(f"Answer:   {response.text}")


# ============================================================
# Step 3: Try different questions
# ============================================================

print("\n" + "=" * 50)
print("Let's try a few more questions...")
print("=" * 50)

questions = [
    "What is Python programming language? One sentence.",
    "Translate 'Hello, how are you?' to Arabic.",
    "What is 15 * 37?",
]

for question in questions:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
    )
    print(f"\nQ: {question}")
    print(f"A: {response.text}")


# ============================================================
# Step 4: Interactive mode — ask your own questions!
# ============================================================

print("\n" + "=" * 50)
print("Your turn! Ask Gemini anything.")
print("Type 'quit' to exit.")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break
    if not user_input:
        continue

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input,
    )
    print(f"Gemini: {response.text}")
