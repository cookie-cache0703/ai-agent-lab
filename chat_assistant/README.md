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
You: What's (18 * 7) + 3?
That's 129.
You: What's the latest news on the Mars rover?
[answer grounded in a live web search]
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

## Tools

Two kinds of tools are wired up, and they plug in differently:

- **Local (client-side) tools** live in `../tools/` (repo root, alongside
  this app). A `Tool` pairs a name/description/pydantic argument schema with
  a handler function; the handler returns either a plain `str` or a
  structured `dict`. `ToolRegistry` turns registered tools into the OpenAI
  function-tool schemas the model sees and dispatches calls back to the
  matching handler; `Agent` JSON-encodes `dict` results before sending them
  back to the model (the API's tool-output field is a string) but keeps the
  raw structured value in the trace. Examples: `time_tool.py`
  (`get_current_time`), `calculator_tool.py` (`calculator`, a safe
  `ast`-based arithmetic evaluator — no `eval`), and `weather_tool.py`
  (`get_weather`, backed by the free Open-Meteo API — no key needed). On
  failure, `get_weather` returns a structured `{"error": ..., "message":
  ...}` dict instead of raising, so the model reads the failure back and can
  explain it to the user instead of the turn crashing. Add a new tool and
  register it in `main.py`'s `build_agent()`.
- **Hosted tools** run on OpenAI's infrastructure — the model calls them and
  gets results back within the same API response, so there's no local
  handler or dispatch step. `main.py`'s `HOSTED_TOOLS` list currently
  enables `web_search`; `Agent` merges these specs in alongside the
  registry's before every model call.

## Tool call trace

Every locally-dispatched tool call (not hosted ones, since those don't expose
per-call arguments/latency) is recorded on `Agent.trace` as a dict:

```json
{"tool_name": "calculator", "arguments": {"expression": "18*7+3"}, "result": "129", "latency_ms": 4}
```

By default each record is printed to stderr as it happens (prefixed with
`[tool call]`). Pass `--trace-file path/to/trace.jsonl` to also append every
record there, one JSON object per line:

```bash
python main.py --trace-file trace.jsonl
```
