"""Nimble Web Search Agent as a native Microsoft Agent Framework ``BaseAgent``."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, Awaitable, Sequence
from typing import Any, Literal, overload

from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    BaseAgent,
    Message,
    ResponseStream,
    normalize_messages,
)
from nimble_python import AsyncNimble

from ._client import (
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_SAFE_READ_RETRIES,
    build_async_client,
    create_generated_run,
    create_run,
    error_from_run,
    fetch_result,
    poll_to_terminal,
)
from ._gating import GatePolicy
from ._mapping import result_to_agent_response
from .exceptions import NimbleInvalidResponseError, NimbleRunOwnershipError

#: Sent as the SDK's ``X-Client-Source`` header via ``client_source=`` on
#: construction -- verified by test_wire_contract.py against the real
#: header the nimble-python client builds, not a hand-maintained constant.
CLIENT_SOURCE = "microsoft-agent-framework"

_STATE_KEY = "nimble"


class NimbleWebSearchAgent(BaseAgent):
    """A Microsoft Agent Framework ``BaseAgent`` backed by Nimble's Agent API V2
    Web Search Agent.

    Two modes, selected by whether ``agent_id`` is configured:

    - **Persistent agent** (``agent_id`` set): every run targets
      ``POST /v2/agents/{agent_id}/runs``. The service's returned
      ``web_search_agent_id`` must equal ``agent_id``; a mismatch raises
      :class:`~nimble_agent_framework.exceptions.NimbleRunOwnershipError`
      rather than being silently accepted.
    - **Generated agent** (``agent_id`` unset): every run targets
      ``POST /v2/agents/runs``. Whichever ``web_search_agent_id`` the
      service returns becomes authoritative for that run's poll/result
      calls, and is cached on the session for later turns.

    Multi-turn conversations are resumed via ``AgentSession``: the Nimble
    ``web_search_agent_id`` and ``interaction_id`` from the most recent run
    are stored in ``session.state["nimble"]`` (and mirrored onto
    ``session.service_session_id``), and the next ``run()`` call on the same
    session sends only the newest message as ``input`` plus
    ``previous_interaction_id`` -- Nimble's own server-side conversation
    state carries the rest.

    ``effort`` is never silently raised: an unset per-run effort omits the
    field entirely so the agent/server default applies, and
    ``effort="max"`` (a coming-soon, custom-budget tier) is rejected by
    default with an actionable message rather than silently sent or
    silently downgraded -- see :mod:`nimble_agent_framework._gating`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_id: str | None = None,
        agent_name: str | None = None,
        name: str | None = None,
        description: str | None = None,
        effort: str | None = None,
        skill: str | None = None,
        use_case: Literal["research", "enrichment", "dataset_building"] | None = None,
        gate_policy: GatePolicy = "reject",
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        safe_read_retries: int = DEFAULT_SAFE_READ_RETRIES,
        client: AsyncNimble | None = None,
        **kwargs: Any,
    ) -> None:
        """Construct the agent.

        Args:
            api_key: Nimble API key. Falls back to the ``NIMBLE_API_KEY``
                environment variable (the SDK's own default) when unset and
                ``client`` is not supplied. Never read back, logged, or
                reflected in any response.
            agent_id: An existing Web Search Agent id (``wsa_...``) to run
                against. Omit to use the generated-agent route instead.
            agent_name: Stable agent name for the generated-agent route
                only; ignored (and rejected) when ``agent_id`` is set.
            name: This ``BaseAgent``'s own framework-level name.
            description: This ``BaseAgent``'s own framework-level description.
            effort: Default per-run effort override (``low``/``medium``/
                ``high``/``x-high``/``max``). Omit to let the server apply
                its own default.
            skill: Default ``skill`` sent with each run.
            use_case: Default ``use_case`` sent with each run.
            gate_policy: How to resolve ``effort="max"`` -- ``"reject"``
                (default) or ``"degrade"`` to ``x-high``.
            base_url: Override the Nimble API base URL.
            timeout_seconds: Per-request HTTP timeout.
            deadline_seconds: Overall poll deadline for a single run.
            poll_interval_seconds: Delay between status polls.
            safe_read_retries: Retry budget for safe, read-only calls only
                (never applied to the non-idempotent create call).
            client: A pre-built ``AsyncNimble`` client (for tests, or to
                share a client across agents). When supplied, ``api_key``/
                ``base_url``/``timeout_seconds`` are ignored.
            **kwargs: Forwarded to ``BaseAgent.__init__`` (``id``,
                ``context_providers``, ``middleware``,
                ``additional_properties``).
        """

        super().__init__(name=name, description=description, **kwargs)
        if agent_id is not None and agent_name is not None:
            raise ValueError(
                "agent_name is ignored on the existing-agent route; configure "
                "either agent_id (persistent agent) or agent_name (generated "
                "agent), not both"
            )
        if client is not None and getattr(client, "max_retries", None) != 0:
            raise ValueError(
                "a pre-built client must be constructed with max_retries=0 "
                "(run creation is billable and non-idempotent; use "
                "nimble_agent_framework._client.build_async_client)"
            )
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._default_effort = effort
        self._skill = skill
        self._use_case = use_case
        self._gate_policy: GatePolicy = gate_policy
        self._deadline_seconds = deadline_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._safe_read_retries = safe_read_retries
        self._client = client or build_async_client(
            api_key=api_key,
            client_source=CLIENT_SOURCE,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    # -- run() overloads (mirrors the documented BaseAgent contract so IDEs
    #    and static type checkers infer the return type from `stream=`) ----

    @overload
    def run(
        self,
        messages: str | Message | Sequence[str | Message] | None = None,
        *,
        stream: Literal[False] = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse]: ...

    @overload
    def run(
        self,
        messages: str | Message | Sequence[str | Message] | None = None,
        *,
        stream: Literal[True],
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse]: ...

    def run(
        self,
        messages: str | Message | Sequence[str | Message] | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse] | ResponseStream[AgentResponseUpdate, AgentResponse]:
        """Execute the agent.

        Extra per-run overrides accepted via ``**kwargs``: ``effort``,
        ``skill``, ``use_case``, ``output_schema``, ``input_data``,
        ``sources``, ``agent_name`` (generated-agent route only),
        ``deadline_seconds``, ``poll_interval_seconds``, and the progress
        observers ``on_created`` (called once with the accepted run) and
        ``on_status`` (called on each status change while polling).
        """

        if stream:
            return ResponseStream(
                self._run_stream(messages=messages, session=session, **kwargs),
                finalizer=AgentResponse.from_updates,
            )
        return self._run(messages=messages, session=session, **kwargs)

    # -- core lifecycle -----------------------------------------------------

    async def _run(
        self,
        messages: str | Message | Sequence[str | Message] | None = None,
        *,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        prompt = self._latest_input_text(messages)
        previous_interaction_id = self._previous_interaction_id(session)

        # Optional one-use admission gate: a sync or async zero-arg callable
        # invoked immediately before the billable, non-idempotent create --
        # before any network request. It must raise to abort the run; a host
        # that needs "only an authorized, single-use grant may create" wires
        # its own admission-claim client here. This hook never runs any
        # network call itself; it is a pure extension point.
        admission_claim = kwargs.get("admission_claim")
        if admission_claim is not None:
            outcome = admission_claim()
            if inspect.isawaitable(outcome):
                await outcome

        created = await self._create(prompt, previous_interaction_id, **kwargs)
        run_id = created.id
        owner_agent_id = self._owning_agent_id(created)

        # Progress observers (optional per-run kwargs): `on_created` fires
        # once with the accepted run (durable id + web_search_agent_id
        # available immediately), `on_status` fires on each status change
        # while polling. Both are observers only -- they cannot alter the
        # lifecycle -- and exist so hosts (e.g. a UI) can show live progress
        # without bypassing the framework-native run() entry point.
        on_created = kwargs.get("on_created")
        if on_created is not None:
            on_created(created)

        terminal = await poll_to_terminal(
            self._client,
            agent_id=owner_agent_id,
            run_id=run_id,
            deadline_seconds=kwargs.get("deadline_seconds", self._deadline_seconds),
            poll_interval_seconds=kwargs.get("poll_interval_seconds", self._poll_interval_seconds),
            safe_read_retries=self._safe_read_retries,
            on_status=kwargs.get("on_status"),
        )
        if terminal.status != "completed":
            raise error_from_run(terminal)

        result = await fetch_result(
            self._client,
            agent_id=owner_agent_id,
            run_id=run_id,
            safe_read_retries=self._safe_read_retries,
        )
        response = result_to_agent_response(result, agent_id=self.id)
        self._remember_session(session, agent_id=owner_agent_id, run=result.run)
        return response

    async def _run_stream(
        self,
        messages: str | Message | Sequence[str | Message] | None = None,
        *,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentResponseUpdate]:
        """Yield the agent's response as a single update.

        Nimble's Agent API V2 is a poll-to-completion research API, not a
        token stream: the adapter only has the final answer once the run
        reaches ``completed``, so there is nothing genuine to emit
        incrementally. Rather than fabricate word-by-word chunking of an
        already-complete answer (which would simulate latency that did not
        happen), this yields exactly one ``AgentResponseUpdate`` carrying
        the full response -- still a correct, spec-conformant
        ``ResponseStream`` producer, and callers that only need the final
        result can still ``await stream.get_final_response()``.
        """

        response = await self._run(messages=messages, session=session, **kwargs)
        for message in response.messages:
            yield AgentResponseUpdate(
                contents=list(message.contents),
                role=message.role,
                agent_id=response.agent_id,
                response_id=response.response_id,
            )

    # -- helpers --------------------------------------------------------

    async def _create(
        self,
        prompt: str,
        previous_interaction_id: str | None,
        **kwargs: Any,
    ) -> Any:
        effort = kwargs.get("effort", self._default_effort)
        skill = kwargs.get("skill", self._skill)
        use_case = kwargs.get("use_case", self._use_case)
        output_schema = kwargs.get("output_schema")
        input_data = kwargs.get("input_data")
        sources = kwargs.get("sources")
        enable_events = kwargs.get("enable_events", False)

        if self._agent_id is not None:
            return await create_run(
                self._client,
                agent_id=self._agent_id,
                prompt=prompt,
                effort=effort,
                gate_policy=self._gate_policy,
                skill=skill,
                use_case=use_case,
                output_schema=output_schema,
                input_data=input_data,
                sources=sources,
                previous_interaction_id=previous_interaction_id,
                # Forwarded so a per-run agent_name is *rejected* by
                # create_run (it is ignored on this route server-side) rather
                # than silently swallowed here.
                agent_name=kwargs.get("agent_name"),
                enable_events=enable_events,
            )
        return await create_generated_run(
            self._client,
            prompt=prompt,
            effort=effort,
            gate_policy=self._gate_policy,
            enable_events=enable_events,
            agent_name=kwargs.get("agent_name", self._agent_name),
            skill=skill,
            use_case=use_case,
            output_schema=output_schema,
            input_data=input_data,
            sources=sources,
            previous_interaction_id=previous_interaction_id,
        )

    def _owning_agent_id(self, created: Any) -> str:
        owner: str | None = created.web_search_agent_id
        if not owner:
            raise NimbleInvalidResponseError(
                "Run creation omitted its owning agent (web_search_agent_id)", run_id=created.id
            )
        if self._agent_id is not None and owner != self._agent_id:
            raise NimbleRunOwnershipError(
                f"Run creation returned owning agent {owner!r}, which does not match "
                f"the configured agent_id {self._agent_id!r}",
                run_id=created.id,
                agent_id=owner,
            )
        return owner

    @staticmethod
    def _latest_input_text(messages: str | Message | Sequence[str | Message] | None) -> str:
        normalized = normalize_messages(messages)
        if not normalized:
            raise ValueError(
                "NimbleWebSearchAgent.run() requires at least one message; "
                "Agent API V2 'input' cannot be empty"
            )
        text = normalized[-1].text
        if not text or not text.strip():
            raise ValueError("The latest message has no text content to send as Nimble 'input'")
        return text

    @staticmethod
    def _previous_interaction_id(session: AgentSession | None) -> str | None:
        if session is None:
            return None
        state = session.state.get(_STATE_KEY)
        if not isinstance(state, dict):
            return None
        interaction_id = state.get("interaction_id")
        return interaction_id if isinstance(interaction_id, str) else None

    @staticmethod
    def _remember_session(session: AgentSession | None, *, agent_id: str, run: Any) -> None:
        if session is None:
            return
        state = session.state.setdefault(_STATE_KEY, {})
        state["web_search_agent_id"] = agent_id
        state["interaction_id"] = run.interaction_id
        state["last_run_id"] = run.id
        session.service_session_id = agent_id
