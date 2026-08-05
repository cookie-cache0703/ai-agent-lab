"""Chat assistant entrypoint."""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import Agent, AgentLimitError
from config import OPENAI_API_KEY
from llm_client import LLMClient, LLMClientError
from tools.calculator_tool import calculator_tool
from tools.registry import ToolRegistry
from tools.search_mock_jobs_tool import search_mock_jobs_tool
from tools.summarize_text_tool import summarize_text_tool
from tools.time_tool import get_current_time_tool
from tools.weather_tool import get_weather_tool

# OpenAI-hosted tools run server-side (no local handler/dispatch needed).
HOSTED_TOOLS = [{"type": "web_search"}]


def make_trace_logger(trace_file: Path | None) -> Callable[[dict], None]:
    def log(record: dict) -> None:
        line = json.dumps(record)
        print(f"[tool call] {line}", file=sys.stderr)
        if trace_file is not None:
            with trace_file.open("a") as f:
                f.write(line + "\n")

    return log


def build_agent(trace_file: Path | None = None) -> Agent:
    tools = ToolRegistry()
    tools.register(get_current_time_tool)
    tools.register(calculator_tool)
    tools.register(get_weather_tool)
    tools.register(search_mock_jobs_tool)
    tools.register(summarize_text_tool)
    return Agent(
        LLMClient(OPENAI_API_KEY),
        tools,
        hosted_tools=HOSTED_TOOLS,
        on_tool_call=make_trace_logger(trace_file),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal chat assistant.")
    parser.add_argument(
        "--trace-file",
        type=Path,
        help="Append each tool call's trace as a JSON line to this file, in addition to printing it.",
    )
    args = parser.parse_args()

    agent = build_agent(trace_file=args.trace_file)

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
        except (LLMClientError, AgentLimitError) as e:
            print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
