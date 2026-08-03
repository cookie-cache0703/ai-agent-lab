"""Chat assistant entrypoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import Agent
from config import OPENAI_API_KEY
from llm_client import LLMClient, LLMClientError
from tools.registry import ToolRegistry
from tools.time_tool import get_current_time_tool


def build_agent() -> Agent:
    tools = ToolRegistry()
    tools.register(get_current_time_tool)
    return Agent(LLMClient(OPENAI_API_KEY), tools)


def main() -> None:
    agent = build_agent()

    print("Chat assistant ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            print(agent.ask(question))
        except LLMClientError as e:
            print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
