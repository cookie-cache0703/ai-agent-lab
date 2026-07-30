"""Chat assistant entrypoint."""

import sys

from config import OPENAI_API_KEY
from llm_client import LLMClient, LLMClientError


def main() -> None:
    question = input("Ask a question: ").strip()
    if not question:
        print("Error: no question provided.", file=sys.stderr)
        sys.exit(1)

    llm_client = LLMClient(OPENAI_API_KEY)
    try:
        response = llm_client.generate(question)
    except LLMClientError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(response)


if __name__ == "__main__":
    main()
