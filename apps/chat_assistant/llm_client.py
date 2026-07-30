"""Thin wrapper around the OpenAI client."""

from openai import APIConnectionError, AuthenticationError, OpenAI, OpenAIError, RateLimitError
from pydantic import BaseModel

from config import config


class Answer(BaseModel):
    answer: str


class LLMClientError(Exception):
    """Raised when an LLM request fails."""


class LLMClient:
    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = config.model

    def generate(self, prompt: str, instructions: str | None = None) -> Answer:
        try:
            response = self.client.with_options(timeout=config.timeout).responses.parse(
                model=self.model,
                instructions=instructions,
                input=prompt,
                temperature=config.temperature,
                text_format=Answer,
            )
        except AuthenticationError:
            raise LLMClientError("invalid OpenAI API key.") from None
        except RateLimitError as e:
            raise LLMClientError(f"rate limit or quota exceeded: {e}") from None
        except APIConnectionError:
            raise LLMClientError("could not reach the OpenAI API. Check your network connection.") from None
        except OpenAIError as e:
            raise LLMClientError(f"OpenAI API request failed: {e}") from None

        return response.output_parsed
