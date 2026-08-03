"""Agent: runs the model/tool-call loop for a single user turn."""

import json

from llm_client import LLMClient
from tools.registry import ToolRegistry


class Agent:
    def __init__(self, llm_client: LLMClient, tools: ToolRegistry) -> None:
        self._llm_client = llm_client
        self._tools = tools
        self._history: list[dict] = []

    def ask(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})

        while True:
            response = self._llm_client.respond(self._history, self._tools.specs())
            self._history.extend(item.model_dump() for item in response.output)

            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                return response.output_text

            for call in function_calls:
                output = self._tools.dispatch(call.name, json.loads(call.arguments))
                self._history.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": output}
                )
