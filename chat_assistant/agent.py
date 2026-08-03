"""Agent: runs the model/tool-call loop for a single user turn."""

import json
import time
from typing import Callable

from llm_client import LLMClient
from tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        hosted_tools: list[dict] | None = None,
        on_tool_call: Callable[[dict], None] | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._tools = tools
        # Hosted tools (e.g. {"type": "web_search"}) run server-side on OpenAI's
        # infrastructure: the model calls them and gets results back within the
        # same response, so they never show up as `function_call` items below.
        self._hosted_tools = hosted_tools or []
        self._on_tool_call = on_tool_call
        self._history: list[dict] = []
        self.trace: list[dict] = []

    def ask(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})

        while True:
            response = self._llm_client.respond(self._history, self._tools.specs() + self._hosted_tools)
            self._history.extend(item.model_dump() for item in response.output)

            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                return response.output_text

            for call in function_calls:
                arguments = json.loads(call.arguments)

                start = time.monotonic()
                result = self._tools.dispatch(call.name, arguments)
                latency_ms = round((time.monotonic() - start) * 1000)

                record = {
                    "tool_name": call.name,
                    "arguments": arguments,
                    "result": result,
                    "latency_ms": latency_ms,
                }
                self.trace.append(record)
                if self._on_tool_call is not None:
                    self._on_tool_call(record)

                output = result if isinstance(result, str) else json.dumps(result)
                self._history.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": output}
                )
