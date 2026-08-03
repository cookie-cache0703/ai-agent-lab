import pytest
from pydantic import BaseModel

from tools.base import Tool
from tools.registry import ToolNotFoundError, ToolRegistry


class _EchoArgs(BaseModel):
    text: str


def _echo(args: _EchoArgs) -> str:
    return args.text


echo_tool = Tool(name="echo", description="Echo text back", args_model=_EchoArgs, handler=_echo)


def test_specs_returns_openai_function_schema_for_registered_tools():
    registry = ToolRegistry()
    registry.register(echo_tool)

    specs = registry.specs()

    assert specs == [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo text back",
            "parameters": _EchoArgs.model_json_schema(),
        }
    ]


def test_dispatch_validates_arguments_and_calls_handler():
    registry = ToolRegistry()
    registry.register(echo_tool)

    result = registry.dispatch("echo", {"text": "hello"})

    assert result == "hello"


def test_dispatch_raises_on_unknown_tool_name():
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError, match="echo"):
        registry.dispatch("echo", {"text": "hello"})
