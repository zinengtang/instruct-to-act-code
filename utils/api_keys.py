"""
API key helpers.

OpenAI key resolution order:
  1. OPENAI_API_KEY environment variable (existing behaviour)
  2. .openai file in the project root (one-line key, no quotes needed)
  3. Raises RuntimeError with a helpful message

Place your key in instruct_to_act/.openai (already in .gitignore).
"""

import os
from pathlib import Path


def load_openai_key() -> str:
    """
    Return the OpenAI API key from the environment or from the .openai file.
    Sets OPENAI_API_KEY in the environment so downstream code that reads it
    directly (e.g. openai.OpenAI()) picks it up automatically.
    """
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if key:
        return key

    # Walk up from this file to find the project root .openai
    candidates = [
        Path(__file__).parent.parent / '.openai',   # instruct_to_act/.openai
        Path(__file__).parent.parent.parent / '.openai',  # project root
        Path.home() / '.openai',
    ]
    for path in candidates:
        if path.exists():
            key = path.read_text().strip()
            if key:
                os.environ['OPENAI_API_KEY'] = key
                return key

    raise RuntimeError(
        "OpenAI API key not found.\n"
        "Set OPENAI_API_KEY in your environment, or place the key in "
        f"{candidates[0]} (one line, no quotes)."
    )
