import json
from unittest.mock import MagicMock

from pydantic import BaseModel

from agent import Agent
from tools.base import Tool
from tools.registry import ToolRegistry


class _FakeOutputItem:
    def __init__(self, type_: str, **fields):
        self.type = type_
        self._fields = fields
        for key, value in fields.items():
            setattr(self, key, value)

    def model_dump(self) -> dict:
        return {"type": self.type, **self._fields}


def _message_response(text: str) -> MagicMock:
    return MagicMock(output=[_FakeOutputItem("message", content=text)], output_text=text)


def _function_call_response(name: str, arguments: dict, call_id: str = "call_1") -> MagicMock:
    item = _FakeOutputItem("function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)
    return MagicMock(output=[item], output_text="")


class _EchoArgs(BaseModel):
    text: str


def _echo(args: _EchoArgs) -> str:
    return args.text


echo_tool = Tool(name="echo", description="Echo text back", args_model=_EchoArgs, handler=_echo)


def test_ask_returns_direct_answer_when_no_tool_call_needed():
    llm_client = MagicMock()
    llm_client.respond.return_value = _message_response("Paris")
    agent = Agent(llm_client, ToolRegistry())

    result = agent.ask("What is the capital of France?")

    assert result == "Paris"
    assert llm_client.respond.call_count == 1


def test_ask_executes_tool_call_and_returns_final_answer():
    llm_client = MagicMock()
    llm_client.respond.side_effect = [
        _function_call_response("echo", {"text": "hi"}, call_id="call_1"),
        _message_response("hi"),
    ]
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = Agent(llm_client, registry)

    result = agent.ask("please echo hi")

    assert result == "hi"
    assert llm_client.respond.call_count == 2

    second_call_history = llm_client.respond.call_args_list[1].args[0]
    assert {"type": "function_call_output", "call_id": "call_1", "output": "hi"} in second_call_history


def test_history_persists_across_ask_calls():
    llm_client = MagicMock()
    llm_client.respond.side_effect = [
        _message_response("Paris"),
        _message_response("Because it's the capital"),
    ]
    agent = Agent(llm_client, ToolRegistry())

    agent.ask("What is the capital of France?")
    agent.ask("Why?")

    # `respond` is called with the same (mutable) history list each time, so by
    # the end of the test it holds the full four-turn conversation.
    final_history = llm_client.respond.call_args_list[1].args[0]
    assert final_history[0] == {"role": "user", "content": "What is the capital of France?"}
    assert final_history[2] == {"role": "user", "content": "Why?"}
    assert len(final_history) == 4
