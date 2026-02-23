"""
Demo: ElevenLabs Voice Cloning
================================
Demonstrates ElevenLabs' voice cloning feature.
Upload a short audio sample of any voice, and ElevenLabs
creates a synthetic version that sounds just like it.

This demo:
  1. Shows how to list your available voices
  2. Generates speech using a specific voice ID
  3. Saves the output as an audio file

Usage:
  python demos/elevenlabs_voice_clone.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

load_dotenv(Path(__file__).parent.parent / ".env")


def list_voices(client: ElevenLabs):
    """List all available voices in your ElevenLabs account."""
    print("🎤 Your Available Voices:")
    print("=" * 50)

    response = client.voices.get_all()
    for voice in response.voices:
        print(f"  Name: {voice.name:20s}  |  ID: {voice.voice_id}")

    print(f"\nTotal: {len(response.voices)} voices")
    return response.voices


def generate_speech(client: ElevenLabs, text: str, voice_id: str, output_path: str):
    """Generate speech using a specific voice and save to file."""
    print(f"\n🔊 Generating speech with voice: {voice_id}")
    print(f"   Text: \"{text[:80]}...\"" if len(text) > 80 else f"   Text: \"{text}\"")

    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_flash_v2_5",
    )

    # Collect all audio chunks and save
    audio_bytes = b"".join(audio_generator)

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    print(f"   Saved: {output_path} ({len(audio_bytes)} bytes)")
    return audio_bytes


def main():
    print("=" * 60)
    print("🎙️  ELEVENLABS VOICE CLONING DEMO")
    print("=" * 60)

    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    # Step 1: List available voices
    voices = list_voices(client)

    if not voices:
        print("\n⚠️  No voices found. Create one at https://elevenlabs.io")
        return

    # Step 2: Generate speech with the first available voice
    voice = voices[0]
    print(f"\n📢 Using voice: {voice.name} ({voice.voice_id})")

    # Arabic sample text
    arabic_text = (
        "مرحبا! أنا صوت مستنسخ باستخدام ElevenLabs. "
        "التكنولوجيا دي بتقدر تستنسخ أي صوت من عينة قصيرة. "
        "تخيل الاحتمالات!"
    )

    # English sample text
    english_text = (
        "Hello! I am a cloned voice powered by ElevenLabs. "
        "This technology can clone any voice from a short sample. "
        "Imagine the possibilities!"
    )

    output_dir = Path(__file__).parent
    generate_speech(client, arabic_text, voice.voice_id, str(output_dir / "output_arabic.mp3"))
    generate_speech(client, english_text, voice.voice_id, str(output_dir / "output_english.mp3"))

    print(f"\n{'=' * 60}")
    print("✅ Voice cloning demo complete!")
    print("   Audio files saved in demos/ directory.")
    print("\n💡 To clone YOUR voice:")
    print("   1. Go to https://elevenlabs.io/voice-cloning")
    print("   2. Upload a 1-minute audio sample of your voice")
    print("   3. Copy the new voice ID")
    print("   4. Use it in your agent's TTS configuration!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
