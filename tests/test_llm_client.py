from unittest.mock import MagicMock

import httpx
import pytest
from openai import AuthenticationError, BadRequestError, RateLimitError, APIConnectionError

from config import LLMConfig
from llm_client import LLMClient, LLMClientError


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/responses")


def _rate_limit_error(msg: str = "rate limited") -> RateLimitError:
    response = httpx.Response(429, request=_request(), json={"error": {"message": msg}})
    return RateLimitError(msg, response=response, body=None)


def _auth_error(msg: str = "invalid api key") -> AuthenticationError:
    response = httpx.Response(401, request=_request(), json={"error": {"message": msg}})
    return AuthenticationError(msg, response=response, body=None)


def _connection_error(msg: str = "connection error") -> APIConnectionError:
    return APIConnectionError(message=msg, request=_request())


def _bad_request_error(msg: str = "bad request") -> BadRequestError:
    response = httpx.Response(400, request=_request(), json={"error": {"message": msg}})
    return BadRequestError(msg, response=response, body=None)


def _make_client() -> tuple[LLMClient, MagicMock]:
    client = LLMClient("test-key")
    client.client = MagicMock()
    mock_create = client.client.with_options.return_value.responses.create
    return client, mock_create


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)


def test_respond_returns_response_on_success():
    client, mock_create = _make_client()
    expected = MagicMock(output_text="Paris")
    mock_create.return_value = expected

    result = client.respond([{"role": "user", "content": "hi"}], tools=[])

    assert result is expected
    assert mock_create.call_count == 1


def test_respond_passes_input_and_tools_through():
    client, mock_create = _make_client()
    mock_create.return_value = MagicMock(output_text="Paris")
    input_items = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "name": "get_current_time"}]

    client.respond(input_items, tools)

    _, kwargs = mock_create.call_args
    assert kwargs["input"] == input_items
    assert kwargs["tools"] == tools


def test_respond_raises_on_authentication_error_without_retrying():
    client, mock_create = _make_client()
    mock_create.side_effect = _auth_error()

    with pytest.raises(LLMClientError, match="invalid OpenAI API key"):
        client.respond([], tools=[])

    assert mock_create.call_count == 1


def test_respond_retries_rate_limit_then_succeeds():
    client, mock_create = _make_client()
    expected = MagicMock(output_text="Paris")
    mock_create.side_effect = [_rate_limit_error(), _rate_limit_error(), expected]

    result = client.respond([], tools=[])

    assert result is expected
    assert mock_create.call_count == 3


def test_respond_retries_connection_error_then_succeeds():
    client, mock_create = _make_client()
    expected = MagicMock(output_text="Paris")
    mock_create.side_effect = [_connection_error(), expected]

    result = client.respond([], tools=[])

    assert result is expected
    assert mock_create.call_count == 2


def test_respond_raises_after_exhausting_retries():
    client, mock_create = _make_client()
    mock_create.side_effect = _rate_limit_error("still limited")

    with pytest.raises(LLMClientError, match="request failed after 3 attempts"):
        client.respond([], tools=[])

    assert mock_create.call_count == 3


def test_respond_raises_on_other_openai_error_without_retrying():
    client, mock_create = _make_client()
    mock_create.side_effect = _bad_request_error()

    with pytest.raises(LLMClientError, match="OpenAI API request failed"):
        client.respond([], tools=[])

    assert mock_create.call_count == 1


def test_llm_config_rejects_max_retries_below_one():
    with pytest.raises(ValueError, match="max_retries must be >= 1"):
        LLMConfig(max_retries=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_tool_rounds": 0}, "max_tool_rounds must be >= 1"),
        ({"max_tool_calls": 0}, "max_tool_calls must be >= 1"),
        ({"max_turn_seconds": 0}, "max_turn_seconds must be > 0"),
    ],
)
def test_llm_config_rejects_invalid_agent_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LLMConfig(**kwargs)
