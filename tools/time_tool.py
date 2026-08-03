"""Example tool: reports the current local date/time."""

from datetime import datetime

from pydantic import BaseModel

from tools.base import Tool


class GetCurrentTimeArgs(BaseModel):
    pass


def _get_current_time(_args: GetCurrentTimeArgs) -> str:
    return datetime.now().isoformat()


get_current_time_tool = Tool(
    name="get_current_time",
    description="Get the current local date and time.",
    args_model=GetCurrentTimeArgs,
    handler=_get_current_time,
)
