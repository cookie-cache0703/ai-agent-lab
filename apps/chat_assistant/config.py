"""Configuration for the chat assistant."""

import os

from dotenv import find_dotenv, load_dotenv

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
load_dotenv(find_dotenv(f".env.{ENVIRONMENT}"))

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

from dataclasses import dataclass


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0
    timeout: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")


config = LLMConfig()