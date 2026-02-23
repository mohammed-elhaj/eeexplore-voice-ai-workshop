"""
Feedback Collection Tool
========================
Collects user feedback about the workshop and saves it to a JSON file.
Each entry includes: name, feedback, timestamp.
"""

import json
from datetime import datetime
from pathlib import Path

# Feedback file path (relative to repo root)
FEEDBACK_FILE = Path(__file__).parent.parent.parent / "feedback_logs" / "feedback.json"


def save_feedback(name: str, feedback: str) -> str:
    """
    Save user feedback to a JSON file.

    Args:
        name: The user's name.
        feedback: The user's feedback comment.

    Returns:
        Confirmation message.
    """
    # Load existing feedback
    entries = []
    if FEEDBACK_FILE.exists():
        try:
            entries = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            entries = []

    # Add new entry
    entry = {
        "name": name,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat(),
    }
    entries.append(entry)

    # Save back to file
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    FEEDBACK_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"📝 Feedback saved! {{'name': '{name}', 'feedback': '{feedback}'}}")
    return f"Feedback from {name} has been saved successfully. Thank you!"
