"""Tool definition shared by the tool registry and any callers of it."""

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], str | dict]

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
        }

    def run(self, arguments: dict) -> str | dict:
        return self.handler(self.args_model.model_validate(arguments))


def tool_error(code: str, message: str) -> dict:
    """Structured error a handler can return (not raise) so the model can explain the failure."""
    return {"error": code, "message": message}
