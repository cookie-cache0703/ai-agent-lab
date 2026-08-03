"""Make the chat_assistant app importable and avoid depending on real secrets."""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "chat_assistant"))
sys.path.insert(0, str(REPO_ROOT))
