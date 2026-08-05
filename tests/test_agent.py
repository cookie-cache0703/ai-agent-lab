import json
from datetime import datetime
from unittest.mock import MagicMock

from pydantic import BaseModel

import pytest

from agent import Agent, AgentLimitError
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


def _raw_function_call_response(name: str, arguments: str, call_id: str = "call_1") -> MagicMock:
    item = _FakeOutputItem("function_call", name=name, arguments=arguments, call_id=call_id)
    return MagicMock(output=[item], output_text="")


class _EchoArgs(BaseModel):
    text: str


def _echo(args: _EchoArgs) -> str:
    return args.text


echo_tool = Tool(name="echo", description="Echo text back", args_model=_EchoArgs, handler=_echo)


class _NoArgs(BaseModel):
    pass


def _structured_result(_args: _NoArgs) -> dict:
    return {"temperature_c": 22.5, "condition": "sunny"}


structured_tool = Tool(
    name="get_weather", description="Fake weather tool", args_model=_NoArgs, handler=_structured_result
)


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


def test_ask_json_encodes_structured_tool_results_for_the_api_but_keeps_them_raw_in_trace():
    llm_client = MagicMock()
    llm_client.respond.side_effect = [
        _function_call_response("get_weather", {}, call_id="call_1"),
        _message_response("It's sunny and 22.5C"),
    ]
    registry = ToolRegistry()
    registry.register(structured_tool)
    agent = Agent(llm_client, registry)

    agent.ask("what's the weather?")

    second_call_history = llm_client.respond.call_args_list[1].args[0]
    function_call_output = next(item for item in second_call_history if item.get("type") == "function_call_output")
    assert function_call_output["output"] == json.dumps({"temperature_c": 22.5, "condition": "sunny"})

    assert agent.trace[0]["result"] == {"temperature_c": 22.5, "condition": "sunny"}


def test_ask_records_a_trace_entry_for_each_tool_call():
    llm_client = MagicMock()
    llm_client.respond.side_effect = [
        _function_call_response("echo", {"text": "hi"}, call_id="call_1"),
        _message_response("hi"),
    ]
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = Agent(llm_client, registry)

    agent.ask("please echo hi")

    assert len(agent.trace) == 1
    record = agent.trace[0]
    assert record["tool_name"] == "echo"
    assert record["arguments"] == {"text": "hi"}
    assert record["result"] == "hi"
    assert isinstance(record["latency_ms"], int)
    assert record["latency_ms"] >= 0


def test_ask_invokes_on_tool_call_callback_with_the_trace_record():
    llm_client = MagicMock()
    llm_client.respond.side_effect = [
        _function_call_response("echo", {"text": "hi"}, call_id="call_1"),
        _message_response("hi"),
    ]
    registry = ToolRegistry()
    registry.register(echo_tool)
    seen = []
    agent = Agent(llm_client, registry, on_tool_call=seen.append)

    agent.ask("please echo hi")

    assert seen == agent.trace


def test_ask_passes_registered_tool_specs_and_hosted_tool_specs_together():
    llm_client = MagicMock()
    llm_client.respond.return_value = _message_response("Paris")
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = Agent(llm_client, registry, hosted_tools=[{"type": "web_search"}])

    agent.ask("search something")

    tools_arg = llm_client.respond.call_args.args[1]
    assert {"type": "web_search"} in tools_arg
    assert any(spec.get("name") == "echo" for spec in tools_arg)


def test_ask_returns_directly_after_a_hosted_tool_call():
    llm_client = MagicMock()
    llm_client.respond.return_value = MagicMock(
        output=[
            _FakeOutputItem("web_search_call", status="completed"),
            _FakeOutputItem("message", content="It's sunny today"),
        ],
        output_text="It's sunny today",
    )
    agent = Agent(llm_client, ToolRegistry(), hosted_tools=[{"type": "web_search"}])

    result = agent.ask("what's the weather today")

    assert result == "It's sunny today"
    assert llm_client.respond.call_count == 1


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


@pytest.mark.parametrize(
    ("first_response", "expected_code"),
    [
        (_raw_function_call_response("echo", "not-json"), "invalid_arguments"),
        (_function_call_response("echo", {}), "invalid_arguments"),
        (_function_call_response("missing", {}), "unknown_tool"),
    ],
)
def test_ask_returns_tool_call_errors_to_the_model(first_response, expected_code):
    llm_client = MagicMock()
    llm_client.respond.side_effect = [first_response, _message_response("I could not use that tool.")]
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = Agent(llm_client, registry)

    result = agent.ask("use a tool")

    assert result == "I could not use that tool."
    second_call_history = llm_client.respond.call_args_list[1].args[0]
    output_item = next(item for item in second_call_history if item.get("type") == "function_call_output")
    assert json.loads(output_item["output"])["error"]["code"] == expected_code
    assert agent.trace[0]["error"]["code"] == expected_code


def test_ask_returns_handler_failure_to_the_model_without_exposing_exception():
    def fail(_args: _NoArgs) -> str:
        raise RuntimeError("secret backend detail")

    failing_tool = Tool(name="fail", description="Fail", args_model=_NoArgs, handler=fail)
    registry = ToolRegistry()
    registry.register(failing_tool)
    llm_client = MagicMock()
    llm_client.respond.side_effect = [
        _function_call_response("fail", {}),
        _message_response("The tool failed."),
    ]
    agent = Agent(llm_client, registry)

    assert agent.ask("fail") == "The tool failed."
    output = next(
        item["output"]
        for item in llm_client.respond.call_args_list[1].args[0]
        if item.get("type") == "function_call_output"
    )
    assert json.loads(output)["error"]["code"] == "tool_execution_failed"
    assert "secret backend detail" not in output


def test_ask_returns_serialization_failure_to_the_model_without_crashing():
    def return_datetime(_args: _NoArgs) -> dict:
        return {"checked_at": datetime(2026, 8, 5, 12, 0)}

    unserializable_tool = Tool(
        name="unserializable",
        description="Return a value that standard JSON cannot encode",
        args_model=_NoArgs,
        handler=return_datetime,
    )
    registry = ToolRegistry()
    registry.register(unserializable_tool)
    llm_client = MagicMock()
    llm_client.respond.side_effect = [
        _function_call_response("unserializable", {}),
        _message_response("The tool result could not be processed."),
    ]
    agent = Agent(llm_client, registry)

    assert agent.ask("use the tool") == "The tool result could not be processed."
    output = next(
        item["output"]
        for item in llm_client.respond.call_args_list[1].args[0]
        if item.get("type") == "function_call_output"
    )
    assert json.loads(output)["error"]["code"] == "tool_execution_failed"
    assert agent.trace[0]["error"]["code"] == "tool_execution_failed"


def test_trace_callback_failure_does_not_leave_tool_call_unresolved():
    llm_client = MagicMock()
    llm_client.respond.side_effect = [
        _function_call_response("echo", {"text": "hi"}),
        _message_response("hi"),
    ]
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = Agent(llm_client, registry, on_tool_call=MagicMock(side_effect=OSError("disk full")))

    assert agent.ask("echo hi") == "hi"
    assert any(
        item.get("type") == "function_call_output"
        for item in llm_client.respond.call_args_list[1].args[0]
    )


def test_ask_stops_after_maximum_tool_rounds_and_resolves_last_call():
    llm_client = MagicMock()
    llm_client.respond.side_effect = [
        _function_call_response("echo", {"text": "one"}, call_id="call_1"),
        _function_call_response("echo", {"text": "two"}, call_id="call_2"),
    ]
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = Agent(llm_client, registry, max_tool_rounds=1)

    with pytest.raises(AgentLimitError, match="tool rounds"):
        agent.ask("loop")

    final_item = llm_client.respond.call_args_list[1].args[0][-1]
    assert final_item["call_id"] == "call_2"
    assert json.loads(final_item["output"])["error"]["code"] == "agent_limit_exceeded"


def test_ask_stops_before_exceeding_maximum_tool_calls():
    llm_client = MagicMock()
    first = MagicMock(
        output=[
            _FakeOutputItem("function_call", name="echo", arguments='{"text":"one"}', call_id="call_1"),
            _FakeOutputItem("function_call", name="echo", arguments='{"text":"two"}', call_id="call_2"),
        ],
        output_text="",
    )
    llm_client.respond.return_value = first
    registry = ToolRegistry()
    registry.register(echo_tool)
    agent = Agent(llm_client, registry, max_tool_calls=1)

    with pytest.raises(AgentLimitError, match="tool calls"):
        agent.ask("call twice")

    assert agent.trace == []
    assert [item["call_id"] for item in llm_client.respond.call_args.args[0][-2:]] == ["call_1", "call_2"]
