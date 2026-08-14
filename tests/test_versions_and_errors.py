"""Version pins and error-taxonomy mapping."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import httpx
import pytest
import tomllib
from conftest import RecordingTransport, json_response, result_body, run_body

from nimble_agent_framework import GatedNimbleEffortError, NimbleWebSearchAgent
from nimble_agent_framework._client import build_async_client
from nimble_agent_framework.exceptions import NimbleAuthError, NimbleInvalidRequestError

_PYPROJECT = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())


def _agent(handler, **kwargs) -> NimbleWebSearchAgent:
    transport = RecordingTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="https://sdk.nimbleway.com")
    client = build_async_client(
        api_key="sk-test-not-a-real-key",
        client_source="microsoft-agent-framework",
        http_client=http_client,
    )
    return NimbleWebSearchAgent(name="Nimble Research", client=client, **kwargs)


def test_installed_versions_satisfy_declared_pins() -> None:
    deps = _PYPROJECT["project"]["dependencies"]
    assert any(d.startswith("agent-framework-core>=1.13.0,<2.0.0") for d in deps)
    assert any(d.startswith("nimble-python>=1.2.0,<2.0.0") for d in deps)

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    for dep in deps:
        name, _, spec = dep.partition(">=")
        pkg = name.strip()
        installed = Version(metadata.version(pkg))
        constraint = SpecifierSet(dep[len(pkg) :])
        assert installed in constraint, f"{pkg}=={installed} does not satisfy {dep}"


async def test_401_maps_to_nimble_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(401, {"message": "invalid API key"})

    agent = _agent(handler)
    with pytest.raises(NimbleAuthError):
        await agent.run("Research this.")


async def test_403_maps_to_nimble_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(403, {"message": "forbidden"})

    agent = _agent(handler)
    with pytest.raises(NimbleAuthError):
        await agent.run("Research this.")


async def test_400_maps_to_nimble_invalid_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(400, {"message": "bad request body"})

    agent = _agent(handler)
    with pytest.raises(NimbleInvalidRequestError):
        await agent.run("Research this.")


async def test_effort_max_raises_gated_error_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the network when effort='max' is rejected")

    agent = _agent(handler)
    with pytest.raises(GatedNimbleEffortError):
        await agent.run("Research this.", effort="max")


async def test_effort_max_degrades_to_x_high_when_agent_opts_in() -> None:
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen_bodies.append(json.loads(request.content))
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler, gate_policy="degrade")
    await agent.run("Research this.", effort="max", poll_interval_seconds=0.001)

    assert seen_bodies[0]["effort"] == "x-high"


async def test_unspecified_effort_is_omitted_from_request_body() -> None:
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen_bodies.append(json.loads(request.content))
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler)
    await agent.run("Research this.", poll_interval_seconds=0.001)

    assert "effort" not in seen_bodies[0]


async def test_output_schema_input_data_sources_pass_through_unchanged() -> None:
    """Structured controls survive both create routes unchanged."""

    seen_bodies: list[dict] = []
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    # Wire-valid Sources shape per nimble-python 1.2 run_create_params:
    # allow/block are lists of {title, domains} groups; avoid/prioritize are
    # free text.
    sources = {
        "allow": [{"title": "Official sources", "domains": ["example.com", "example.org"]}],
        "avoid": "low-quality aggregators",
    }
    input_data = [{"name": "Acme Corp"}]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen_bodies.append(json.loads(request.content))
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body(output_type="json", content={"summary": "ok"}))
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler)
    await agent.run(
        "Enrich this record.",
        output_schema=schema,
        sources=sources,
        input_data=input_data,
        use_case="enrichment",
        poll_interval_seconds=0.001,
    )

    body = seen_bodies[0]
    assert body["output_schema"] == schema
    assert body["sources"] == sources
    assert body["input_data"] == input_data
    assert body["use_case"] == "enrichment"


async def test_skill_use_case_agent_name_are_typed_params_not_extra_body() -> None:
    """skill/use_case/agent_name travel as named JSON fields on the
    generated route -- never nested under an extra_body escape hatch."""

    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen_bodies.append(json.loads(request.content))
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        return json_response(200, run_body(status="completed"))

    agent = _agent(
        handler,
        agent_name="stable-research-agent",
        skill="competitive-intelligence",
        use_case="research",
    )
    await agent.run("Research this.", poll_interval_seconds=0.001)

    body = seen_bodies[0]
    assert body["agent_name"] == "stable-research-agent"
    assert body["skill"] == "competitive-intelligence"
    assert body["use_case"] == "research"
    assert "extra_body" not in body


async def test_invalid_use_case_rejected_client_side() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the network with an invalid use_case")

    agent = _agent(handler)
    with pytest.raises(ValueError, match="not a supported Web Search Agent use case"):
        await agent.run("Research this.", use_case="not-a-real-use-case")


async def test_429_maps_to_rate_limit_error_and_500_to_server_error() -> None:
    """Retryable-after-backoff (429) and ambiguous-outcome (5xx) failures
    must be distinguishable from permanently-wrong requests."""

    from nimble_agent_framework import NimbleRateLimitError, NimbleServerError

    def handler_429(request: httpx.Request) -> httpx.Response:
        return json_response(429, {"message": "rate limited"})

    agent = _agent(handler_429)
    with pytest.raises(NimbleRateLimitError):
        await agent.run("Research this.")

    def handler_500(request: httpx.Request) -> httpx.Response:
        return json_response(500, {"message": "server error"})

    agent = _agent(handler_500)
    with pytest.raises(NimbleServerError):
        await agent.run("Research this.")


async def test_error_str_is_clean_message_even_with_inner_exception() -> None:
    """str(exc) must be the human message, never a rendered args tuple --
    it flows directly into host UIs."""

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(401, {"message": "invalid API key"})

    agent = _agent(handler)
    with pytest.raises(NimbleAuthError) as excinfo:
        await agent.run("Research this.")

    text = str(excinfo.value)
    assert not text.startswith("(")
    assert "AuthenticationError(" not in text
    assert excinfo.value.__cause__ is not None  # inner exception preserved for debugging


async def test_all_adapter_errors_share_the_public_base() -> None:
    from nimble_agent_framework import (
        NimbleAgentError,
        NimbleAuthError,
        NimbleInvalidRequestError,
        NimbleInvalidResponseError,
        NimbleRateLimitError,
        NimbleRunFailedError,
        NimbleRunOwnershipError,
        NimbleRunTimeoutError,
        NimbleServerError,
    )

    for cls in (
        NimbleAuthError,
        NimbleInvalidRequestError,
        NimbleInvalidResponseError,
        NimbleRateLimitError,
        NimbleRunFailedError,
        NimbleRunOwnershipError,
        NimbleRunTimeoutError,
        NimbleServerError,
    ):
        assert issubclass(cls, NimbleAgentError)


async def test_agent_name_rejected_via_public_api_on_persistent_route() -> None:
    """A per-run agent_name on the /{agent_id}/runs route is rejected loudly
    through the public run() surface, never silently swallowed."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the network")

    from conftest import AGENT_ID

    agent = _agent(handler, agent_id=AGENT_ID)
    with pytest.raises(ValueError, match="ignored on the existing-agent route"):
        await agent.run("Research this.", agent_name="should-be-rejected")


def test_constructor_rejects_agent_id_plus_agent_name() -> None:
    with pytest.raises(ValueError, match="not both"):
        NimbleWebSearchAgent(
            name="x", api_key="sk-test", agent_id="wsa_1", agent_name="also-a-name"
        )


def test_constructor_rejects_prebuilt_client_with_retries_enabled() -> None:
    from nimble_python import AsyncNimble

    retrying_client = AsyncNimble(api_key="sk-test", max_retries=2)
    with pytest.raises(ValueError, match="max_retries=0"):
        NimbleWebSearchAgent(name="x", client=retrying_client)


async def test_safe_read_retry_budget_recovers_transient_poll_failure() -> None:
    """The bounded safe-read budget really does retry a transient 500 on the
    read-only status GET -- while the create POST stays single-attempt."""


    get_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_attempts
        if request.method == "POST":
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        get_attempts += 1
        if get_attempts == 1:
            return json_response(500, {"message": "transient blip"})
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler)
    response = await agent.run("Research this.", poll_interval_seconds=0.001)

    assert response.messages[-1].text
    assert get_attempts == 2  # first attempt 500, retried once, succeeded
