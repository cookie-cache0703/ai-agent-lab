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


config = LLMConfig()