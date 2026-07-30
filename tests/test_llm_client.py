from unittest.mock import MagicMock

import httpx
import pytest
from openai import AuthenticationError, BadRequestError, RateLimitError, APIConnectionError

from config import LLMConfig
from llm_client import Answer, LLMClient, LLMClientError


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
    mock_parse = client.client.with_options.return_value.responses.parse
    return client, mock_parse


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)


def test_generate_returns_parsed_answer_on_success():
    client, mock_parse = _make_client()
    mock_parse.return_value = MagicMock(output_parsed=Answer(answer="Paris"))

    result = client.generate("What is the capital of France?")

    assert result == Answer(answer="Paris")
    assert mock_parse.call_count == 1


def test_generate_raises_on_authentication_error_without_retrying():
    client, mock_parse = _make_client()
    mock_parse.side_effect = _auth_error()

    with pytest.raises(LLMClientError, match="invalid OpenAI API key"):
        client.generate("hi")

    assert mock_parse.call_count == 1


def test_generate_retries_rate_limit_then_succeeds():
    client, mock_parse = _make_client()
    mock_parse.side_effect = [
        _rate_limit_error(),
        _rate_limit_error(),
        MagicMock(output_parsed=Answer(answer="Paris")),
    ]

    result = client.generate("hi")

    assert result == Answer(answer="Paris")
    assert mock_parse.call_count == 3


def test_generate_retries_connection_error_then_succeeds():
    client, mock_parse = _make_client()
    mock_parse.side_effect = [
        _connection_error(),
        MagicMock(output_parsed=Answer(answer="Paris")),
    ]

    result = client.generate("hi")

    assert result == Answer(answer="Paris")
    assert mock_parse.call_count == 2


def test_generate_raises_after_exhausting_retries():
    client, mock_parse = _make_client()
    mock_parse.side_effect = _rate_limit_error("still limited")

    with pytest.raises(LLMClientError, match="request failed after 3 attempts"):
        client.generate("hi")

    assert mock_parse.call_count == 3


def test_generate_raises_on_other_openai_error_without_retrying():
    client, mock_parse = _make_client()
    mock_parse.side_effect = _bad_request_error()

    with pytest.raises(LLMClientError, match="OpenAI API request failed"):
        client.generate("hi")

    assert mock_parse.call_count == 1


def test_llm_config_rejects_max_retries_below_one():
    with pytest.raises(ValueError, match="max_retries must be >= 1"):
        LLMConfig(max_retries=0)
