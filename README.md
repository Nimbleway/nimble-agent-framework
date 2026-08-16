# nimble-agent-framework

A native [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
`BaseAgent` backed by [Nimble](https://nimbleway.com)'s Agent API V2 Web
Search Agent — the same async create/poll/result research-agent lifecycle
Nimble's other framework integrations use, wrapped as a first-class
`agent_framework.BaseAgent` you can drop into any Agent Framework
application.

> **Status:** early release. The core lifecycle, effort gating, session
> resumability, and result-fidelity behavior described below are covered by
> unit tests; live characterization against the real Nimble API is ongoing.

## Install

```bash
pip install nimble-agent-framework
```

Requires Python ≥3.10, `agent-framework-core>=1.13.0,<2.0.0`, and
`nimble-python>=1.2.0,<2.0.0` (installed automatically as dependencies).

## Quick start

```python
import asyncio
from nimble_agent_framework import NimbleWebSearchAgent

async def main() -> None:
    # Reads NIMBLE_API_KEY from the environment if api_key is not passed.
    agent = NimbleWebSearchAgent(name="Nimble Research")
    response = await agent.run("What changed in the latest stable release of the Python requests library?")
    print(response.messages[-1].text)

asyncio.run(main())
```

## Two agent modes

```python
# Generated agent (default): each run creates/reuses an agent server-side.
agent = NimbleWebSearchAgent(name="Nimble Research")

# Persistent agent: run against an account agent you already created.
agent = NimbleWebSearchAgent(name="Nimble Research", agent_id="wsa_...")
```

## Multi-turn sessions

```python
session = agent.create_session()
first = await agent.run("Research recent news about the Python packaging ecosystem.", session=session)
# The next call continues the same Nimble conversation server-side via
# previous_interaction_id, stored on `session.state["nimble"]`.
followup = await agent.run("Now summarize just the security-relevant changes.", session=session)
```

## Streaming

```python
stream = agent.run("Summarize the current state of PEP 723.", stream=True, session=session)
async for update in stream:
    print(update.text)
final_response = await stream.get_final_response()
```

Nimble's Agent API V2 is a poll-to-completion research API, not a token
stream — the adapter only has the final answer once the run reaches
`completed`, so `stream=True` yields exactly one `AgentResponseUpdate`
carrying the whole answer rather than fabricating word-by-word chunking of
an already-complete result. `stream=True` is still useful for code that's
already written against the framework's streaming contract.

## Effort tiers

```python
# Omit `effort` to let Nimble apply its own server-side default.
await agent.run("...", effort="high")

# effort="max" is not generally available. It is rejected by
# default with an actionable error rather than silently sent or downgraded:
await agent.run("...", effort="max")
# nimble_agent_framework.GatedNimbleEffortError: effort='max' is a
# not generally available and is not sent to Nimble. Pass effort='x-high' to
# proceed now, or construct NimbleWebSearchAgent with gate_policy='degrade'
# to opt into automatic substitution of 'x-high'.

agent = NimbleWebSearchAgent(name="Nimble Research", gate_policy="degrade")
await agent.run("...", effort="max")  # silently proceeds at effort="x-high"
```

## Result fidelity

The completed run's text/JSON output and full trust metadata (confidence,
reasoning, sources, per-claim citations with excerpts) are preserved as
native Python objects — never flattened into a string — on
`content.additional_properties["nimble"]`:

```python
response = await agent.run("...")
content = response.messages[-1].contents[0]
nimble = content.additional_properties["nimble"]
nimble["output_type"]   # "text" | "json"
nimble["output"]        # the raw text or structured JSON, untouched
nimble["trust"]         # {"confidence", "reasoning", "sources": [...], "claims": [...]}
nimble["run"]           # {"id", "web_search_agent_id", "effort", "interaction_id", "status", ...}
```

For `output_type == "json"` runs, `response.value` also carries the raw
structured output directly (the field the framework defines for exactly
this purpose).

## Configuration

| Constructor arg | Default | Notes |
|---|---|---|
| `api_key` | `NIMBLE_API_KEY` env var | Never logged or reflected in errors |
| `agent_id` | `None` | Persistent-agent mode when set |
| `agent_name` | `None` | Generated-agent route only; rejected if `agent_id` is set |
| `effort` | `None` (server default) | `low`/`medium`/`high`/`x-high`/`max` |
| `gate_policy` | `"reject"` | `"reject"` \| `"degrade"` for `effort="max"` |
| `skill` / `use_case` | `None` | Passed through to the SDK's typed 1.2 fields |
| `deadline_seconds` | `300.0` | Overall poll deadline per run |
| `poll_interval_seconds` | `10.0` | Delay between status polls |
| `safe_read_retries` | `2` | Retry budget for GET calls only — the create call always uses `max_retries=0` |

Per-run overrides for `effort`, `skill`, `use_case`, `output_schema`,
`input_data`, `sources`, `agent_name`, `deadline_seconds`,
`poll_interval_seconds` are accepted as `**kwargs` on `run()`.

## Known limitations (V1)

- No pass-through of Nimble's own SSE run-progress events
  (`enable_events=True` → `stream_events()`); framework-level `stream=True`
  is supported by yielding the completed answer as a single update (see
  "Streaming" above).
- `input_data`/enrichment/`dataset_building` typed parameters are threaded
  through the API, but this release's own tests/demo focus on the
  `research` use case.

## Contributing

Issues and pull requests are welcome. Run `pip install -e ".[dev]"`, then
`pytest`, `ruff check .`, and `mypy src` before submitting.

## License

MIT — see [`LICENSE`](LICENSE).

Microsoft and Microsoft Agent Framework are trademarks of Microsoft Corporation.
This project is independently developed by Nimbleway and is not endorsed by or
affiliated with Microsoft.

## Security

API keys are read from `NIMBLE_API_KEY` (or the `api_key` constructor
argument) and passed only as the SDK's own `Authorization: Bearer ...`
header. No key value is ever logged, included in an exception message, or
reflected in a response.
