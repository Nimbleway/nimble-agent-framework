"""Nimble Agent API V2 lifecycle helpers.

This module only imports ``nimble_python`` and this package's own
``_gating``/``exceptions`` modules -- it has no dependency on
``agent_framework``'s ``Message``/``Content``/``AgentResponse`` mapping types
-- so the create/poll/result lifecycle discipline (zero create retries,
bounded polling, typed 1.2 parameters, owner-id preservation) can be unit
tested in isolation from the framework response-mapping layer in
:mod:`nimble_agent_framework._mapping` and :mod:`nimble_agent_framework.agent`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

import nimble_python
from nimble_python import AsyncNimble, omit

from ._gating import GatePolicy, resolve_effort
from .exceptions import (
    NimbleAuthError,
    NimbleInvalidRequestError,
    NimbleInvalidResponseError,
    NimbleRateLimitError,
    NimbleRunFailedError,
    NimbleRunTimeoutError,
    NimbleServerError,
)

#: Poll every 10s by default, configurable, with a 300s overall deadline.
DEFAULT_DEADLINE_SECONDS = 300.0
DEFAULT_POLL_INTERVAL_SECONDS = 10.0

#: Retry budget for safe, read-only calls only (GET .../runs/{id} and
#: .../result). Never applied to the billable, non-idempotent create call.
DEFAULT_SAFE_READ_RETRIES = 2

TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_STATES = frozenset({"queued", "running"})

_USE_CASES = frozenset({"research", "enrichment", "dataset_building"})


def _typed(value: Any | None) -> Any:
    """Send a typed value, or the SDK's ``omit`` sentinel when unset.

    Passing ``None`` would serialize an explicit null; ``omit`` leaves the
    field out of the request body entirely so the server-side default (an
    agent's stored effort, an existing agent's ``use_case``, ...) applies.
    """

    return omit if value is None else cast(Any, value)


def _checked_use_case(use_case: str | None) -> str | None:
    if use_case is not None and use_case not in _USE_CASES:
        raise ValueError(
            f"use_case={use_case!r} is not a supported Web Search Agent use case "
            f"(expected one of {sorted(_USE_CASES)!r})"
        )
    return use_case


def build_async_client(
    *,
    api_key: str | None = None,
    client_source: str,
    base_url: str | None = None,
    timeout_seconds: float = 30.0,
    http_client: Any | None = None,
) -> AsyncNimble:
    """Construct the SDK client with ``max_retries=0``.

    Run creation is billable and non-idempotent, and the Agent API V2
    contract exposes no idempotency key, so the *client-wide* default is
    zero retries. Safe, read-only calls opt back into a bounded retry budget
    per call via :func:`safe_reader` / ``client.with_options(...)``, never
    the reverse.
    """

    kwargs: dict[str, Any] = {
        "client_source": client_source,
        "timeout": timeout_seconds,
        "max_retries": 0,
    }
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    if http_client is not None:
        kwargs["http_client"] = http_client
    return AsyncNimble(**kwargs)


def safe_reader(client: AsyncNimble, *, retries: int = DEFAULT_SAFE_READ_RETRIES) -> AsyncNimble:
    """A view of ``client`` that retries only safe, read-only calls."""

    return client.with_options(max_retries=retries)


def _map_api_error(exc: Exception, *, agent_id: str | None, run_id: str | None = None) -> Exception:
    """Map a nimble-python exception to this adapter's typed taxonomy.

    The distinctions matter because create is never auto-retried: a caller
    building its own retry policy must be able to tell "the request is
    permanently wrong" (:class:`NimbleInvalidRequestError`) apart from
    "well-formed but throttled" (:class:`NimbleRateLimitError`) and
    "service-side failure, outcome possibly ambiguous"
    (:class:`NimbleServerError`).

    ``run_id`` is ``None`` for the create-call sites (no run exists yet) but
    must be passed by any *post-create* caller (polling, result fetch) --
    otherwise a caller catching the resulting typed exception has no way to
    identify or recover the already-created, already-billed run, and may
    resubmit a duplicate.
    """

    if isinstance(exc, nimble_python.AuthenticationError | nimble_python.PermissionDeniedError):
        return NimbleAuthError(str(exc), run_id=run_id, agent_id=agent_id, inner_exception=exc)
    if isinstance(exc, nimble_python.RateLimitError):
        return NimbleRateLimitError(str(exc), run_id=run_id, agent_id=agent_id, inner_exception=exc)
    if isinstance(exc, nimble_python.APIStatusError):
        if exc.status_code >= 500:
            return NimbleServerError(str(exc), run_id=run_id, agent_id=agent_id, inner_exception=exc)
        return NimbleInvalidRequestError(str(exc), run_id=run_id, agent_id=agent_id, inner_exception=exc)
    if isinstance(exc, nimble_python.APIConnectionError):
        return NimbleInvalidResponseError(
            f"Nimble request failed: {exc}", run_id=run_id, agent_id=agent_id, inner_exception=exc
        )
    return exc


async def create_run(
    client: AsyncNimble,
    *,
    agent_id: str,
    prompt: str,
    effort: str | None = None,
    gate_policy: GatePolicy = "reject",
    skill: str | None = None,
    use_case: str | None = None,
    output_schema: dict[str, Any] | None = None,
    input_data: dict[str, Any] | list[dict[str, Any]] | None = None,
    sources: dict[str, Any] | None = None,
    previous_interaction_id: str | None = None,
    agent_name: str | None = None,
    enable_events: bool = False,
) -> Any:
    """Start a run on an existing account agent (``POST /v2/agents/{agent_id}/runs``).

    Exactly one HTTP attempt: ``client`` must have been built with
    ``max_retries=0`` (see :func:`build_async_client`). ``agent_name`` is
    deliberately rejected here rather than silently dropped: the published
    contract says it is ignored on this route, so accepting it would imply
    an effect that does not happen. ``enable_events`` is forwarded as-is
    (default ``False``); this module never consumes the resulting SSE
    stream -- a caller that sets it is responsible for calling
    ``client.agents.runs.stream_events`` separately.
    """

    if agent_name is not None:
        raise ValueError(
            "agent_name is ignored on the existing-agent route "
            "(POST /v2/agents/{agent_id}/runs); use the generated-agent "
            "route (no agent_id configured) to name or reuse an agent by name"
        )
    if not prompt or not prompt.strip():
        raise ValueError("prompt/input is required")
    resolved_effort = resolve_effort(effort, gate_policy=gate_policy)
    resolved_use_case = _checked_use_case(use_case)
    try:
        return await client.agents.runs.create(
            agent_id,
            input=prompt,
            effort=_typed(resolved_effort),
            skill=_typed(skill),
            use_case=_typed(resolved_use_case),
            output_schema=_typed(output_schema),
            input_data=_typed(input_data),
            sources=_typed(sources),
            previous_interaction_id=_typed(previous_interaction_id),
            enable_events=enable_events,
        )
    except Exception as exc:
        raise _map_api_error(exc, agent_id=agent_id) from exc


async def create_generated_run(
    client: AsyncNimble,
    *,
    prompt: str,
    effort: str | None = None,
    gate_policy: GatePolicy = "reject",
    agent_name: str | None = None,
    skill: str | None = None,
    use_case: str | None = None,
    output_schema: dict[str, Any] | None = None,
    input_data: dict[str, Any] | list[dict[str, Any]] | None = None,
    sources: dict[str, Any] | None = None,
    previous_interaction_id: str | None = None,
    enable_events: bool = False,
) -> Any:
    """Start a run on the no-agent-id route (``POST /v2/agents/runs``).

    An unseen ``agent_name`` creates a new agent; an existing one reuses it.
    Exactly one HTTP attempt, for the same reason as :func:`create_run`.
    """

    if not prompt or not prompt.strip():
        raise ValueError("prompt/input is required")
    resolved_effort = resolve_effort(effort, gate_policy=gate_policy)
    resolved_use_case = _checked_use_case(use_case)
    try:
        return await client.agents.run(
            input=prompt,
            effort=_typed(resolved_effort),
            agent_name=_typed(agent_name),
            skill=_typed(skill),
            use_case=_typed(resolved_use_case),
            output_schema=_typed(output_schema),
            input_data=_typed(input_data),
            sources=_typed(sources),
            previous_interaction_id=_typed(previous_interaction_id),
            enable_events=enable_events,
        )
    except Exception as exc:
        raise _map_api_error(exc, agent_id=None) from exc


async def poll_to_terminal(
    client: AsyncNimble,
    *,
    agent_id: str,
    run_id: str,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    safe_read_retries: int = DEFAULT_SAFE_READ_RETRIES,
    on_status: Callable[[str], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Any:
    """Poll ``GET .../runs/{run_id}`` to a terminal state.

    Only ``queued``/``running`` (active) and ``completed``/``failed``/
    ``cancelled`` (terminal) are recognized; any other value is a typed
    protocol error that retains ``run_id``/``agent_id`` rather than being
    silently treated as active.
    """

    reader = safe_reader(client, retries=safe_read_retries)
    deadline = monotonic() + deadline_seconds
    previous_status: str | None = None
    while True:
        try:
            run = await reader.agents.runs.get(run_id, agent_id=agent_id)
        except Exception as exc:
            raise _map_api_error(exc, agent_id=agent_id, run_id=run_id) from exc
        status = run.status
        if status != previous_status and on_status is not None:
            # Observer-only: the run is already accepted and being polled
            # server-side, so a raise here (a UI panel disposed, a
            # telemetry sink down) must not stop polling -- otherwise
            # unrelated observer code would strand an in-flight run.
            with contextlib.suppress(Exception):
                on_status(status)
        previous_status = status
        if status in TERMINAL_STATES:
            return run
        if status not in ACTIVE_STATES:
            raise NimbleInvalidResponseError(
                f"Unknown Nimble run status {status!r}", run_id=run_id, agent_id=agent_id
            )
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise NimbleRunTimeoutError(
                "Timed out while the Nimble run was still active", run_id=run_id, agent_id=agent_id
            )
        await sleep(min(poll_interval_seconds, remaining))


def error_from_run(run: Any) -> NimbleRunFailedError:
    """Build a :class:`NimbleRunFailedError` from a terminal ``failed``/``cancelled`` run.

    Used when :func:`poll_to_terminal` itself reports a non-``completed``
    terminal status: ``GET .../runs/{run_id}`` already carries the run's
    ``error`` field in that case, so there is no need to additionally call
    ``/result`` (which would just 422 with the same information).
    """

    error = run.error
    message = error.message if error is not None else "Nimble run ended without a completed result"
    return NimbleRunFailedError(
        f"Nimble run ended without a completed result: {message}",
        run_id=run.id,
        agent_id=run.web_search_agent_id,
        status=run.status,
        ref_id=error.ref_id if error is not None else None,
    )


async def fetch_result(
    client: AsyncNimble,
    *,
    agent_id: str,
    run_id: str,
    safe_read_retries: int = DEFAULT_SAFE_READ_RETRIES,
) -> Any:
    """Fetch ``GET .../runs/{run_id}/result``.

    Only call this after :func:`poll_to_terminal` reports ``completed`` --
    the documented ``409`` (still active) and ``422`` (failed/cancelled)
    boundaries are handled here defensively for the rare race where the
    service's state moves between the status check and this call, not as
    the primary control flow.
    """

    reader = safe_reader(client, retries=safe_read_retries)
    try:
        return await reader.agents.runs.result(run_id, agent_id=agent_id)
    except nimble_python.ConflictError as exc:
        raise NimbleInvalidResponseError(
            "Nimble result was not ready (409) even though the run had reported a terminal state",
            run_id=run_id,
            agent_id=agent_id,
            inner_exception=exc,
        ) from exc
    except nimble_python.UnprocessableEntityError as exc:
        raise _failed_result_error(exc, run_id=run_id, agent_id=agent_id) from exc
    except Exception as exc:
        raise _map_api_error(exc, agent_id=agent_id, run_id=run_id) from exc


def _failed_result_error(
    exc: nimble_python.UnprocessableEntityError, *, run_id: str, agent_id: str
) -> NimbleRunFailedError:
    """Build a :class:`NimbleRunFailedError` from a 422 ``TaskRunFailedResultPublicV2`` body.

    ``exc.body`` is the parsed JSON body -- ``{"error": {...}, "run": {...}}``
    -- when the API returned valid JSON; fall back to the exception's own
    message if the body is missing or an unexpected shape (never crash on a
    malformed error body).
    """

    body: dict[str, Any] = exc.body if isinstance(exc.body, dict) else {}
    raw_error = body.get("error")
    error: dict[str, Any] = raw_error if isinstance(raw_error, dict) else {}
    raw_run = body.get("run")
    run: dict[str, Any] = raw_run if isinstance(raw_run, dict) else {}
    message = error.get("message") or str(exc)
    return NimbleRunFailedError(
        f"Nimble run ended without a completed result: {message}",
        run_id=run_id,
        agent_id=agent_id,
        status=run.get("status"),
        ref_id=error.get("ref_id"),
        details=body,
    )
