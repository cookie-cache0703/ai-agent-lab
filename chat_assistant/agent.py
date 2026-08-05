"""Agent: runs the model/tool-call loop for a single user turn."""

import json
import time
from typing import Any, Callable

from pydantic import ValidationError

from config import config
from llm_client import LLMClient
from tools.registry import ToolNotFoundError, ToolRegistry


class AgentLimitError(Exception):
    """Raised when a user turn exceeds its configured execution budget."""


class Agent:
    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        hosted_tools: list[dict] | None = None,
        on_tool_call: Callable[[dict], None] | None = None,
        max_tool_rounds: int | None = None,
        max_tool_calls: int | None = None,
        max_turn_seconds: float | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._tools = tools
        # Hosted tools (e.g. {"type": "web_search"}) run server-side on OpenAI's
        # infrastructure: the model calls them and gets results back within the
        # same response, so they never show up as `function_call` items below.
        self._hosted_tools = hosted_tools or []
        self._on_tool_call = on_tool_call
        self._max_tool_rounds = max_tool_rounds or config.max_tool_rounds
        self._max_tool_calls = max_tool_calls or config.max_tool_calls
        self._max_turn_seconds = max_turn_seconds or config.max_turn_seconds
        self._history: list[dict] = []
        self.trace: list[dict] = []

    def ask(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})
        turn_started = time.monotonic()
        tool_rounds = 0
        tool_calls_made = 0

        while True:
            if time.monotonic() - turn_started >= self._max_turn_seconds:
                raise AgentLimitError("turn exceeded the maximum elapsed time")

            response = self._llm_client.respond(self._history, self._tools.specs() + self._hosted_tools)
            self._history.extend(item.model_dump() for item in response.output)

            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                return response.output_text

            tool_rounds += 1
            if tool_rounds > self._max_tool_rounds:
                self._append_limit_outputs(function_calls, "maximum tool rounds exceeded")
                raise AgentLimitError("turn exceeded the maximum number of tool rounds")
            if tool_calls_made + len(function_calls) > self._max_tool_calls:
                self._append_limit_outputs(function_calls, "maximum tool calls exceeded")
                raise AgentLimitError("turn exceeded the maximum number of tool calls")

            for call in function_calls:
                start = time.monotonic()
                arguments, result, output, error = self._execute_tool_call(call.name, call.arguments)
                latency_ms = round((time.monotonic() - start) * 1000)
                tool_calls_made += 1

                record = {
                    "tool_name": call.name,
                    "arguments": arguments,
                    "result": result,
                    "latency_ms": latency_ms,
                }
                if error is not None:
                    record["error"] = error
                self.trace.append(record)
                if self._on_tool_call is not None:
                    try:
                        self._on_tool_call(record)
                    except Exception:
                        # Observability must never prevent a tool result from being
                        # returned to the model and leave the call unresolved.
                        pass

                self._history.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": output}
                )

    def _execute_tool_call(
        self, name: str, raw_arguments: str
    ) -> tuple[dict, Any, str, dict | None]:
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            error = {"code": "invalid_arguments", "message": str(exc), "retryable": True}
            result = {"error": error}
            return {}, result, json.dumps(result), error

        try:
            result = self._tools.dispatch(name, arguments)
            output = result if isinstance(result, str) else json.dumps(result)
            return arguments, result, output, None
        except ValidationError as exc:
            error = {
                "code": "invalid_arguments",
                "message": "Arguments did not match the tool schema.",
                "details": exc.errors(include_url=False, include_input=False),
                "retryable": True,
            }
        except ToolNotFoundError:
            error = {"code": "unknown_tool", "message": f"Tool {name!r} is not available.", "retryable": False}
        except Exception:
            # Do not expose handler or serialization exception details to the
            # model or traces.
            error = {"code": "tool_execution_failed", "message": "The tool failed while executing.", "retryable": True}
        result = {"error": error}
        return arguments, result, json.dumps(result), error

    def _append_limit_outputs(self, calls: list, message: str) -> None:
        output = json.dumps(
            {"error": {"code": "agent_limit_exceeded", "message": message, "retryable": False}}
        )
        for call in calls:
            self._history.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": output}
            )
