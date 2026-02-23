"""
Session 1: Prompt Engineering — Interactive Comparison
======================================================
This script demonstrates how different prompts produce dramatically
different outputs from the same LLM. It's the "Manager's Skill" —
learning to communicate effectively with AI.

Three techniques:
  1. Give it a Persona — "Act as a Senior Engineer"
  2. Define your Audience — "Explain to a 5-year-old"
  3. Add Constraints — "Using a food analogy, in 2 sentences"

Usage:
  python session1_basics/03_prompt_engineering.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent.parent / ".env")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the response."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def compare_prompts(topic: str):
    """Compare a lazy prompt vs an engineered prompt on the same topic."""
    print(f"\n{'=' * 60}")
    print(f"📝 TOPIC: {topic}")
    print(f"{'=' * 60}")

    # Lazy prompt
    lazy_prompt = f"Tell me about {topic}"
    print(f"\n❌ LAZY PROMPT: \"{lazy_prompt}\"")
    print("-" * 40)
    lazy_response = ask_gemini(lazy_prompt)
    print(lazy_response[:500])

    # Engineered prompt
    engineered_prompt = (
        f"Act as a Senior AI Engineer explaining to university students "
        f"who are beginners. Explain {topic} using a simple real-world analogy. "
        f"Keep it under 3 sentences. Be engaging and conversational."
    )
    print(f"\n✅ ENGINEERED PROMPT: \"{engineered_prompt}\"")
    print("-" * 40)
    engineered_response = ask_gemini(engineered_prompt)
    print(engineered_response[:500])


def voice_prompt_demo():
    """Show the difference between chat prompts and voice prompts."""
    print(f"\n{'=' * 60}")
    print(f"🎤 VOICE AI PROMPTING — Chat vs Voice")
    print(f"{'=' * 60}")

    topic = "how neural networks learn"

    # Chat-style prompt (bad for voice)
    chat_prompt = f"Explain {topic} in detail with bullet points and examples."
    print(f"\n❌ CHAT PROMPT (bad for voice): \"{chat_prompt}\"")
    print("-" * 40)
    chat_response = ask_gemini(chat_prompt)
    print(chat_response[:600])
    print("\n⚠️  This works for reading, but sounds TERRIBLE when spoken aloud!")

    # Voice-optimized prompt
    voice_prompt = (
        f"You are a voice assistant. The user will HEAR your response, not read it. "
        f"Explain {topic} in 2-3 short, natural sentences. "
        f"Use conversational language. No bullet points, no formatting, no lists. "
        f"Speak as if you're talking to a friend."
    )
    print(f"\n✅ VOICE PROMPT (optimized): \"{voice_prompt}\"")
    print("-" * 40)
    voice_response = ask_gemini(voice_prompt)
    print(voice_response[:500])
    print("\n✅ This sounds natural when spoken aloud!")


def interactive_mode():
    """Let the user experiment with their own prompts."""
    print(f"\n{'=' * 60}")
    print(f"🧪 YOUR TURN — Experiment with prompts!")
    print(f"{'=' * 60}")
    print("Type a prompt and see the response. Type 'quit' to exit.\n")

    while True:
        prompt = input("Your prompt: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        print("-" * 40)
        response = ask_gemini(prompt)
        print(response)
        print()


def main():
    print("🎓 PROMPT ENGINEERING — The Manager's Skill")
    print("=" * 60)
    print("The quality of AI output depends 100% on YOUR input.")
    print("Let's see the difference between lazy and engineered prompts.\n")

    # Demo 1: Compare prompts on LLMs
    compare_prompts("Large Language Models (LLMs)")

    # Demo 2: Compare prompts on voice AI
    compare_prompts("voice AI agents")

    # Demo 3: Chat vs Voice prompting
    voice_prompt_demo()

    # Demo 4: Interactive mode
    try:
        interactive_mode()
    except KeyboardInterrupt:
        pass

    print("\n🎯 KEY TAKEAWAY:")
    print("   Give it a Persona + Define your Audience + Add Constraints")
    print("   = Dramatically better results every time.\n")


if __name__ == "__main__":
    main()
