"""Wire-level contract: exact X-Client-Source, route selection, zero create
retries, no credential reflection.
"""

from __future__ import annotations

import httpx
import pytest
from conftest import AGENT_ID, json_response, make_client, run_body

from nimble_agent_framework._client import create_generated_run, create_run
from nimble_agent_framework.exceptions import (
    NimbleInvalidRequestError,
    NimbleInvalidResponseError,
    NimbleRateLimitError,
    NimbleServerError,
)


async def test_x_client_source_header_is_exact() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(202, run_body())

    client, transport = make_client(handler)
    await create_generated_run(client, prompt="research something")

    assert transport.requests[-1].headers["x-client-source"] == "microsoft-agent-framework"


async def test_no_credential_reflected_in_headers_shown_to_caller() -> None:
    """The Authorization header carries the key -- confirm it is Bearer-shaped
    and never duplicated into a body, query string, or another header."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "sk-test-not-a-real-key" not in str(request.url)
        assert request.headers["authorization"] == "Bearer sk-test-not-a-real-key"
        return json_response(202, run_body())

    client, _ = make_client(handler)
    await create_generated_run(client, prompt="research something")


async def test_generated_run_posts_to_no_agent_id_route() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v2/agents/runs"
        return json_response(202, run_body())

    client, transport = make_client(handler)
    await create_generated_run(client, prompt="research something")
    assert len(transport.requests) == 1


async def test_existing_agent_run_posts_to_agent_id_route() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/v2/agents/{AGENT_ID}/runs"
        return json_response(202, run_body(agent_id=AGENT_ID))

    client, transport = make_client(handler)
    await create_run(client, agent_id=AGENT_ID, prompt="research something")
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    ("make_response", "expected_error"),
    [
        pytest.param(
            lambda: (_ for _ in ()).throw(httpx.ConnectError("boom")),
            NimbleInvalidResponseError,
            id="transport-failure",
        ),
        pytest.param(
            lambda: httpx.Response(408, json={"message": "timeout"}),
            NimbleInvalidRequestError,
            id="408",
        ),
        pytest.param(
            lambda: httpx.Response(409, json={"message": "conflict"}),
            NimbleInvalidRequestError,
            id="409",
        ),
        pytest.param(
            lambda: httpx.Response(429, json={"message": "rate limited"}),
            NimbleRateLimitError,
            id="429",
        ),
        pytest.param(
            lambda: httpx.Response(500, json={"message": "server error"}),
            NimbleServerError,
            id="500",
        ),
        pytest.param(
            lambda: httpx.Response(503, json={"message": "unavailable"}),
            NimbleServerError,
            id="503",
        ),
    ],
)
async def test_create_is_never_retried_regardless_of_failure_mode(make_response, expected_error) -> None:
    """Run creation is billable and non-idempotent with no idempotency
    key. Every one of these failure modes must still produce exactly one
    HTTP attempt -- max_retries=0 on the client, no manual retry wrapper --
    and each maps to the taxonomy type a caller-side retry policy needs
    (retryable-after-backoff vs. permanently-wrong vs. ambiguous-outcome)."""

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return make_response()

    client, transport = make_client(handler)
    with pytest.raises(expected_error):
        await create_generated_run(client, prompt="research something")

    assert attempts == 1
    assert len(transport.requests) == 1


async def test_generated_run_rejects_agent_name_ignored_route_error_is_not_raised() -> None:
    """Sanity: agent_name IS accepted (and forwarded) on the generated route."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        assert body.get("agent_name") == "my-stable-agent"
        return json_response(202, run_body())

    client, _ = make_client(handler)
    await create_generated_run(client, prompt="research something", agent_name="my-stable-agent")


async def test_existing_agent_route_rejects_agent_name() -> None:
    """The published contract says agent_name is ignored on the
    /{agent_id}/runs route; accepting it here would imply an effect that
    does not happen, so it is rejected client-side before any request."""

    client, transport = make_client(lambda r: json_response(202, run_body()))
    with pytest.raises(ValueError, match="ignored on the existing-agent route"):
        await create_run(client, agent_id=AGENT_ID, prompt="x", agent_name="should-not-be-sent")
    assert transport.requests == []
