"""Make the chat_assistant app importable and avoid depending on real secrets."""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

CHAT_ASSISTANT_DIR = Path(__file__).resolve().parent.parent / "apps" / "chat_assistant"
sys.path.insert(0, str(CHAT_ASSISTANT_DIR))
