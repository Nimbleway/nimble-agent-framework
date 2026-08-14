"""Framework-level streaming (`run(stream=True)` -> ResponseStream)."""

from __future__ import annotations

import httpx
from agent_framework import AgentResponseUpdate
from conftest import AGENT_ID, RecordingTransport, json_response, result_body, run_body

from nimble_agent_framework import NimbleWebSearchAgent
from nimble_agent_framework._client import build_async_client


def _agent(handler) -> NimbleWebSearchAgent:
    transport = RecordingTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="https://sdk.nimbleway.com")
    client = build_async_client(
        api_key="sk-test-not-a-real-key",
        client_source="microsoft-agent-framework",
        http_client=http_client,
    )
    return NimbleWebSearchAgent(name="Nimble Research", client=client)


def _handler(request: httpx.Request) -> httpx.Response:
    if request.method == "POST":
        return json_response(202, run_body(status="queued"))
    if request.url.path.endswith("/result"):
        return json_response(200, result_body(content="Streamed answer.[1]"))
    return json_response(200, run_body(status="completed"))


async def test_stream_true_returns_response_stream_with_one_update() -> None:
    agent = _agent(_handler)
    stream = agent.run("Research this.", stream=True, poll_interval_seconds=0.001)

    updates: list[AgentResponseUpdate] = []
    async for update in stream:
        updates.append(update)

    assert len(updates) == 1
    assert updates[0].text == "Streamed answer.[1]"
    assert updates[0].agent_id == agent.id


async def test_stream_true_get_final_response_preserves_nimble_metadata() -> None:
    agent = _agent(_handler)
    stream = agent.run("Research this.", stream=True, poll_interval_seconds=0.001)

    async for _ in stream:
        pass
    final = await stream.get_final_response()

    assert final.messages[-1].text == "Streamed answer.[1]"
    nimble = final.messages[-1].contents[0].additional_properties["nimble"]
    assert nimble["run"]["web_search_agent_id"] == AGENT_ID
    assert nimble["trust"]["confidence"] == "high"


async def test_stream_true_get_final_response_preserves_structured_value() -> None:
    """Verify that a stream=True JSON result does not lose
    AgentResponse.value in get_final_response() -- the non-streaming route
    (_run) and the streaming route (_run_stream) must agree on .value for
    the same underlying result."""

    def json_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(
                200, result_body(output_type="json", content={"summary": "ok"})
            )
        return json_response(200, run_body(status="completed"))

    agent = _agent(json_handler)
    stream = agent.run("Enrich this.", stream=True, poll_interval_seconds=0.001)

    async for _ in stream:
        pass
    final = await stream.get_final_response()

    assert final.value == {"summary": "ok"}
