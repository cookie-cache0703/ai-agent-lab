"""Thin wrapper around the OpenAI client."""

import time

from openai import APIConnectionError, AuthenticationError, OpenAI, OpenAIError, RateLimitError
from pydantic import BaseModel

from config import config


class Answer(BaseModel):
    answer: str


class LLMClientError(Exception):
    """Raised when an LLM request fails."""


class LLMClient:
    def __init__(self, api_key: str) -> None:
        # max_retries=0: retries for transient errors are handled explicitly in generate().
        self.client = OpenAI(api_key=api_key, max_retries=0)
        self.model = config.model

    def generate(self, prompt: str, instructions: str | None = None) -> Answer:
        for attempt in range(1, config.max_retries + 1):
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
            except (RateLimitError, APIConnectionError) as e:
                if attempt == config.max_retries:
                    raise LLMClientError(f"request failed after {config.max_retries} attempts: {e}") from None
                time.sleep(config.retry_backoff_seconds * 2 ** (attempt - 1))
                continue
            except OpenAIError as e:
                raise LLMClientError(f"OpenAI API request failed: {e}") from None

            return response.output_parsed
