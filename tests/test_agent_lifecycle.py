"""End-to-end NimbleWebSearchAgent lifecycle over a mocked transport."""

from __future__ import annotations

import httpx
import pytest
from conftest import AGENT_ID, RUN_ID, RecordingTransport, json_response, result_body, run_body

from nimble_agent_framework import NimbleWebSearchAgent
from nimble_agent_framework._client import (
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    build_async_client,
    poll_to_terminal,
)
from nimble_agent_framework.exceptions import (
    NimbleInvalidResponseError,
    NimbleRunFailedError,
    NimbleRunOwnershipError,
    NimbleRunTimeoutError,
)


def _agent(handler, *, agent_id: str | None = None) -> NimbleWebSearchAgent:
    transport = RecordingTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="https://sdk.nimbleway.com")
    client = build_async_client(
        api_key="sk-test-not-a-real-key",
        client_source="microsoft-agent-framework",
        http_client=http_client,
    )
    return NimbleWebSearchAgent(name="Nimble Research", agent_id=agent_id, client=client)


def _router(routes: dict[tuple[str, str], list[httpx.Response] | httpx.Response]):
    calls: dict[tuple[str, str], int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        calls[key] = calls.get(key, 0) + 1
        entry = routes[key]
        if isinstance(entry, list):
            index = min(calls[key] - 1, len(entry) - 1)
            return entry[index]
        return entry

    return handler, calls


async def test_generated_run_completes_and_preserves_returned_agent_id() -> None:
    """Generated route; whichever web_search_agent_id the service
    returns is authoritative for poll/result, not a configured fallback."""

    returned_agent_id = "wsa_generated_by_service"
    handler, calls = _router(
        {
            ("POST", "/v2/agents/runs"): json_response(
                202, run_body(status="queued", agent_id=returned_agent_id)
            ),
            ("GET", f"/v2/agents/{returned_agent_id}/runs/{RUN_ID}"): json_response(
                200, run_body(status="completed", agent_id=returned_agent_id)
            ),
            ("GET", f"/v2/agents/{returned_agent_id}/runs/{RUN_ID}/result"): json_response(
                200, result_body(agent_id=returned_agent_id)
            ),
        }
    )
    agent = _agent(handler)
    response = await agent.run("What changed recently?", poll_interval_seconds=0.001)

    assert response.messages[-1].text
    nimble = response.messages[-1].contents[0].additional_properties["nimble"]
    assert nimble["run"]["web_search_agent_id"] == returned_agent_id
    assert calls[("POST", "/v2/agents/runs")] == 1


async def test_existing_agent_run_uses_agent_id_route_and_matches_owner() -> None:
    """Configured agent_id routes to /{agent_id}/runs; a matching
    returned web_search_agent_id is accepted."""

    handler, calls = _router(
        {
            ("POST", f"/v2/agents/{AGENT_ID}/runs"): json_response(202, run_body(status="queued")),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}"): json_response(
                200, run_body(status="completed")
            ),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}/result"): json_response(200, result_body()),
        }
    )
    agent = _agent(handler, agent_id=AGENT_ID)
    response = await agent.run("Research this.", poll_interval_seconds=0.001)

    assert response.messages[-1].text
    assert calls[("POST", f"/v2/agents/{AGENT_ID}/runs")] == 1


async def test_existing_agent_run_id_mismatch_is_rejected() -> None:
    """Never silently substitute the configured agent_id for a
    different owner the service actually returned."""

    handler, _ = _router(
        {
            ("POST", f"/v2/agents/{AGENT_ID}/runs"): json_response(
                202, run_body(status="queued", agent_id="wsa_a_totally_different_agent")
            ),
        }
    )
    agent = _agent(handler, agent_id=AGENT_ID)
    with pytest.raises(NimbleRunOwnershipError) as excinfo:
        await agent.run("Research this.")
    assert excinfo.value.agent_id == "wsa_a_totally_different_agent"


async def test_lifecycle_transitions_queued_running_completed() -> None:
    """queued -> running -> completed is a valid lifecycle."""

    handler, calls = _router(
        {
            ("POST", "/v2/agents/runs"): json_response(202, run_body(status="queued")),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}"): [
                json_response(200, run_body(status="queued")),
                json_response(200, run_body(status="running")),
                json_response(200, run_body(status="completed")),
            ],
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}/result"): json_response(200, result_body()),
        }
    )
    agent = _agent(handler)
    response = await agent.run("Research this.", poll_interval_seconds=0.001)

    assert response.messages[-1].text
    assert calls[("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}")] == 3


async def test_failed_run_raises_with_run_and_agent_id() -> None:
    handler, _ = _router(
        {
            ("POST", "/v2/agents/runs"): json_response(202, run_body(status="queued")),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}"): json_response(
                200,
                run_body(status="failed", error={"message": "no relevant sources found", "ref_id": RUN_ID}),
            ),
        }
    )
    agent = _agent(handler)
    with pytest.raises(NimbleRunFailedError) as excinfo:
        await agent.run("Research this.", poll_interval_seconds=0.001)

    assert excinfo.value.run_id == RUN_ID
    assert excinfo.value.agent_id == AGENT_ID
    assert excinfo.value.status == "failed"
    assert "no relevant sources found" in str(excinfo.value)


async def test_cancelled_run_raises_run_failed_error() -> None:
    handler, _ = _router(
        {
            ("POST", "/v2/agents/runs"): json_response(202, run_body(status="queued")),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}"): json_response(
                200, run_body(status="cancelled", error={"message": "cancelled by caller", "ref_id": RUN_ID})
            ),
        }
    )
    agent = _agent(handler)
    with pytest.raises(NimbleRunFailedError) as excinfo:
        await agent.run("Research this.", poll_interval_seconds=0.001)
    assert excinfo.value.status == "cancelled"


async def test_unknown_status_raises_typed_protocol_error_with_ids() -> None:
    handler, _ = _router(
        {
            ("POST", "/v2/agents/runs"): json_response(202, run_body(status="queued")),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}"): json_response(
                200, {**run_body(status="queued"), "status": "somehow_new_status"}
            ),
        }
    )
    agent = _agent(handler)
    with pytest.raises(NimbleInvalidResponseError) as excinfo:
        await agent.run("Research this.", poll_interval_seconds=0.001)
    assert excinfo.value.run_id == RUN_ID
    assert excinfo.value.agent_id == AGENT_ID
    assert "somehow_new_status" in str(excinfo.value)


async def test_result_409_while_technically_still_active_is_handled() -> None:
    """The documented 409 boundary (result not ready) is handled as
    data, not a crash -- even in the rare race where our own status check
    already said 'completed' but /result briefly still 409s."""

    handler, _ = _router(
        {
            ("POST", "/v2/agents/runs"): json_response(202, run_body(status="queued")),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}"): json_response(
                200, run_body(status="completed")
            ),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}/result"): json_response(
                409, {"message": "run result not ready", "ref_id": RUN_ID}
            ),
        }
    )
    agent = _agent(handler)
    with pytest.raises(NimbleInvalidResponseError) as excinfo:
        await agent.run("Research this.", poll_interval_seconds=0.001)
    assert excinfo.value.run_id == RUN_ID


async def test_result_422_after_status_says_completed_maps_to_run_failed_error() -> None:
    """The documented 422 boundary carries a structured {error, run}
    body that must not be discarded."""

    handler, _ = _router(
        {
            ("POST", "/v2/agents/runs"): json_response(202, run_body(status="queued")),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}"): json_response(
                200, run_body(status="completed")
            ),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}/result"): json_response(
                422,
                {
                    "error": {"message": "compression failed", "ref_id": RUN_ID},
                    "run": run_body(status="failed"),
                },
            ),
        }
    )
    agent = _agent(handler)
    with pytest.raises(NimbleRunFailedError) as excinfo:
        await agent.run("Research this.", poll_interval_seconds=0.001)
    assert "compression failed" in str(excinfo.value)
    assert excinfo.value.ref_id == RUN_ID
    assert excinfo.value.details["error"]["message"] == "compression failed"


async def test_default_poll_interval_and_deadline_match_launch_contract() -> None:
    assert DEFAULT_POLL_INTERVAL_SECONDS == 10.0
    assert DEFAULT_DEADLINE_SECONDS == 300.0


async def test_poll_interval_is_configurable_without_real_sleeping() -> None:
    """Default 10s, but configurable -- proven with a fake clock, no
    real sleep, so the test suite stays fast."""

    statuses = iter(["queued", "running", "completed"])
    sleeps: list[float] = []

    class _Run:
        def __init__(self, status: str) -> None:
            self.status = status
            self.id = RUN_ID
            self.web_search_agent_id = AGENT_ID

    class _Reader:
        async def get(self, run_id: str, *, agent_id: str) -> _Run:
            return _Run(next(statuses))

    class _RunsNamespace:
        def __init__(self, reader: _Reader) -> None:
            self.runs = reader

    class _FakeClient:
        def __init__(self) -> None:
            self.agents = _RunsNamespace(_Reader())

        def with_options(self, *, max_retries: int) -> _FakeClient:
            return self

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    fake_time = {"t": 0.0}

    def fake_monotonic() -> float:
        return fake_time["t"]

    result = await poll_to_terminal(
        _FakeClient(),
        agent_id=AGENT_ID,
        run_id=RUN_ID,
        poll_interval_seconds=25.0,
        deadline_seconds=1000.0,
        sleep=fake_sleep,
        monotonic=fake_monotonic,
    )

    assert result.status == "completed"
    assert sleeps == [25.0, 25.0]


async def test_poll_timeout_raises_with_run_and_agent_id() -> None:
    fake_time = {"t": 0.0}

    def fake_monotonic() -> float:
        return fake_time["t"]

    async def fake_sleep(seconds: float) -> None:
        fake_time["t"] += seconds + 1.0  # always exceed the deadline

    class _Run:
        status = "running"
        id = RUN_ID
        web_search_agent_id = AGENT_ID

    class _Reader:
        async def get(self, run_id: str, *, agent_id: str) -> _Run:
            return _Run()

    class _RunsNamespace:
        runs = _Reader()

    class _FakeClient:
        agents = _RunsNamespace()

        def with_options(self, *, max_retries: int) -> _FakeClient:
            return self

    with pytest.raises(NimbleRunTimeoutError) as excinfo:
        await poll_to_terminal(
            _FakeClient(),
            agent_id=AGENT_ID,
            run_id=RUN_ID,
            poll_interval_seconds=10.0,
            deadline_seconds=5.0,
            sleep=fake_sleep,
            monotonic=fake_monotonic,
        )
    assert excinfo.value.run_id == RUN_ID
    assert excinfo.value.agent_id == AGENT_ID


async def test_x_client_source_present_through_full_agent_lifecycle() -> None:
    seen_headers: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        if request.method == "POST":
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler)
    await agent.run("Research this.", poll_interval_seconds=0.001)

    assert seen_headers
    assert all(h["x-client-source"] == "microsoft-agent-framework" for h in seen_headers)


async def test_on_created_and_on_status_observers_fire_during_run() -> None:
    """Progress observers surface durable IDs and each status transition
    while run() is in flight -- the hook a UI would build a live progress
    indicator on."""

    handler, _ = _router(
        {
            ("POST", "/v2/agents/runs"): json_response(202, run_body(status="queued")),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}"): [
                json_response(200, run_body(status="queued")),
                json_response(200, run_body(status="running")),
                json_response(200, run_body(status="completed")),
            ],
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}/result"): json_response(200, result_body()),
        }
    )
    agent = _agent(handler)
    created_runs: list = []
    statuses: list[str] = []

    await agent.run(
        "Research this.",
        poll_interval_seconds=0.001,
        on_created=created_runs.append,
        on_status=statuses.append,
    )

    assert len(created_runs) == 1
    assert created_runs[0].id == RUN_ID
    assert created_runs[0].web_search_agent_id == AGENT_ID
    assert statuses == ["queued", "running", "completed"]


async def test_admission_claim_hook_runs_before_any_network_request() -> None:
    """A sync admission_claim callable that raises must abort before the
    create POST -- zero requests reach the transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the network when admission is denied")

    agent = _agent(handler)

    def deny() -> None:
        raise RuntimeError("admission denied")

    with pytest.raises(RuntimeError, match="admission denied"):
        await agent.run("Research this.", admission_claim=deny)


async def test_async_admission_claim_hook_is_awaited_before_create() -> None:
    calls: list[str] = []

    handler, _ = _router(
        {
            ("POST", "/v2/agents/runs"): json_response(202, run_body(status="queued")),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}"): json_response(
                200, run_body(status="completed")
            ),
            ("GET", f"/v2/agents/{AGENT_ID}/runs/{RUN_ID}/result"): json_response(200, result_body()),
        }
    )
    agent = _agent(handler)

    async def admit() -> None:
        calls.append("admission")

    response = await agent.run(
        "Research this.", poll_interval_seconds=0.001, admission_claim=admit
    )

    assert calls == ["admission"]
    assert response.messages[-1].text


async def test_enable_events_flag_reaches_the_create_body() -> None:
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        if request.method == "POST":
            seen_bodies.append(_json.loads(request.content))
            return json_response(202, run_body(status="queued"))
        if request.url.path.endswith("/result"):
            return json_response(200, result_body())
        return json_response(200, run_body(status="completed"))

    agent = _agent(handler)
    await agent.run("Research this.", enable_events=True, poll_interval_seconds=0.001)

    assert seen_bodies[0]["enable_events"] is True
