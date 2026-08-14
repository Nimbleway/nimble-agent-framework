"""Resumable multi-turn sessions via AgentSession.

Preserves agent_id / run_id / interaction_id across turns and threads
previous_interaction_id server-side rather than resending full history.
"""

from __future__ import annotations

import json

import httpx
from agent_framework import AgentSession
from conftest import AGENT_ID, RUN_ID, RecordingTransport, json_response, result_body, run_body

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


async def test_first_run_has_no_previous_interaction_id() -> None:
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen_bodies.append(json.loads(request.content))
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler)
    session = agent.create_session()
    await agent.run("First question.", session=session, poll_interval_seconds=0.001)

    assert "previous_interaction_id" not in seen_bodies[0]


async def test_session_state_and_service_session_id_populated_after_run() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler)
    session = agent.create_session()
    await agent.run("First question.", session=session, poll_interval_seconds=0.001)

    nimble_state = session.state["nimble"]
    assert nimble_state["web_search_agent_id"] == AGENT_ID
    assert nimble_state["interaction_id"] == "interaction_1"
    assert nimble_state["last_run_id"] == RUN_ID
    assert session.service_session_id == AGENT_ID


async def test_second_run_on_same_session_sends_previous_interaction_id_and_only_new_message() -> None:
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen_bodies.append(json.loads(request.content))
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler)
    session = agent.create_session()
    await agent.run("First question.", session=session, poll_interval_seconds=0.001)
    await agent.run("Now narrow it down.", session=session, poll_interval_seconds=0.001)

    assert len(seen_bodies) == 2
    assert "previous_interaction_id" not in seen_bodies[0]
    assert seen_bodies[1]["previous_interaction_id"] == "interaction_1"
    # Only the newest turn's text is sent; the server-side interaction
    # carries the earlier turn, so the first question is not resent.
    assert seen_bodies[1]["input"] == "Now narrow it down."
    assert "First question" not in seen_bodies[1]["input"]


async def test_run_without_a_session_omits_previous_interaction_id_every_time() -> None:
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen_bodies.append(json.loads(request.content))
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler)
    await agent.run("First question.", poll_interval_seconds=0.001)
    await agent.run("Second, unrelated question.", poll_interval_seconds=0.001)

    assert "previous_interaction_id" not in seen_bodies[0]
    assert "previous_interaction_id" not in seen_bodies[1]


async def test_session_round_trips_through_to_dict_from_dict() -> None:
    """AgentSession's own serialization contract must survive our state dict
    (plain JSON-able values only -- no SDK model objects leaked into it)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler)
    session = agent.create_session()
    await agent.run("First question.", session=session, poll_interval_seconds=0.001)

    serialized = session.to_dict()
    restored = AgentSession.from_dict(serialized)
    assert restored.state["nimble"]["interaction_id"] == "interaction_1"
    assert restored.service_session_id == AGENT_ID
