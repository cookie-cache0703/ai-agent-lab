"""Registry mapping tool names to their definitions for lookup and dispatch."""

from tools.base import Tool


class ToolNotFoundError(Exception):
    """Raised when dispatch() is called with a name that isn't registered."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def specs(self) -> list[dict]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def dispatch(self, name: str, arguments: dict) -> str:
        try:
            tool = self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"no tool registered with name {name!r}") from None
        return tool.run(arguments)
