# Chat Assistant

A terminal chat assistant backed by the OpenAI Responses API. The model
decides on each turn whether it can answer directly or needs to call a tool;
if a tool is needed, the assistant runs it locally and feeds the result back
to the model before returning a final answer.

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

This starts a REPL: keep asking questions in the same session (conversation
history is kept in memory so follow-ups work), and type `exit`, `quit`, or
send EOF (Ctrl-D) to stop.

```
Chat assistant ready. Type 'exit' or 'quit' to stop.
You: What is the capital of France?
The capital of France is Paris.
You: What time is it right now?
The current time is 2026-08-03T09:14:02.
You: exit
```

## Structure

- `main.py` — entrypoint; runs the REPL loop and wires up the `Agent` with
  the available tools.
- `agent.py` — `Agent` holds the conversation history and runs the
  model/tool-call loop for each `ask()`: call the model, execute any
  requested tool calls via the registry, feed the results back, and repeat
  until the model returns a final answer.
- `llm_client.py` — `LLMClient` wraps the OpenAI client, applying the
  configured model/temperature/timeout, translating OpenAI errors into a
  single `LLMClientError`, and retrying transient failures.
- `config.py` — loads the environment-specific `.env` file and exposes
  `OPENAI_API_KEY` plus an `LLMConfig` (`model`, `temperature`, `timeout`).

Tool definitions live in `../tools/` (repo root, alongside this app): a
`Tool` pairs a name/description/pydantic argument schema with a
handler function, and `ToolRegistry` turns registered tools into the OpenAI
function-tool schemas the model sees and dispatches calls back to the
matching handler. `tools/time_tool.py` is a minimal example tool
(`get_current_time`) — add new tools there and register them in
`main.py`'s `build_agent()`.
