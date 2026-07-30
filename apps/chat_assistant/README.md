# Chat Assistant

A minimal terminal chat assistant backed by the OpenAI Responses API.

## Setup

1. Install dependencies (from the repo root):

   ```bash
   pip install -r requirements.txt
   ```

2. Add your OpenAI API key to the appropriate env file at the repo root
   (`.env.local`, `.env.development`, `.env.staging`, or `.env.production`):

   ```
   OPENAI_API_KEY=sk-...
   ```

   Which file is loaded is controlled by the `ENVIRONMENT` env var
   (defaults to `local`).

## Usage

```bash
python main.py
```

You'll be prompted to type a question in the terminal; the assistant's
reply is printed back.

```
Ask a question: What is the capital of France?
The capital of France is Paris.
```

## Structure

- `main.py` — entrypoint; reads the question from the terminal and prints
  the response.
- `llm_client.py` — `LLMClient` wraps the OpenAI client, applying the
  configured model/temperature/timeout and translating OpenAI errors into
  a single `LLMClientError`.
- `config.py` — loads the environment-specific `.env` file and exposes
  `OPENAI_API_KEY` plus an `LLMConfig` (`model`, `temperature`, `timeout`).
