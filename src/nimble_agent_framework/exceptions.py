"""Typed exceptions for the Nimble Web Search Agent adapter.

All exceptions descend from ``agent_framework.exceptions.IntegrationException``,
the branch that framework's own ``CODING_STANDARD.md`` names for "non-chat
external dependencies like search services or custom APIs" -- as opposed to
``AgentException`` (agent-level logic) or ``ChatClientException`` (LLM
provider protocol issues), neither of which fits a Web Search Agent backend.
"""

from __future__ import annotations

from typing import Any

from agent_framework.exceptions import (
    IntegrationException,
    IntegrationInvalidAuthException,
    IntegrationInvalidRequestException,
    IntegrationInvalidResponseException,
)

__all__ = [
    "NimbleAgentError",
    "NimbleAuthError",
    "NimbleInvalidRequestError",
    "NimbleInvalidResponseError",
    "NimbleRateLimitError",
    "NimbleRunFailedError",
    "NimbleRunTimeoutError",
    "NimbleRunOwnershipError",
    "NimbleServerError",
]


class _NimbleErrorMixin:
    """Adds Nimble run/agent identifiers to an ``IntegrationException`` subclass."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        agent_id: str | None = None,
        inner_exception: Exception | None = None,
    ) -> None:
        super().__init__(message, inner_exception)  # type: ignore[call-arg]
        # The framework base forwards inner_exception into Exception.args;
        # with len(args) != 1 Python's default __str__ renders the whole args
        # tuple instead of the message. Normalize so str(exc) is always the
        # clean message (the inner exception stays on __cause__/context).
        self.args = (message,)
        self.run_id = run_id
        self.agent_id = agent_id


class NimbleAgentError(_NimbleErrorMixin, IntegrationException):
    """Base class for every error raised by the Nimble Web Search Agent adapter.

    ``except NimbleAgentError`` catches the whole adapter taxonomy; the
    subclasses additionally participate in the framework's more specific
    ``IntegrationInvalid*Exception`` branches where one applies.
    """


class NimbleAuthError(NimbleAgentError, IntegrationInvalidAuthException):
    """The Nimble API rejected the request's credentials (401/403)."""


class NimbleInvalidRequestError(NimbleAgentError, IntegrationInvalidRequestException):
    """The Nimble API rejected the request itself (400/404/409/422 status codes).

    Retrying the same request without changing it will not help.
    """


class NimbleRateLimitError(NimbleAgentError):
    """The Nimble API rate-limited the request (429).

    Distinct from :class:`NimbleInvalidRequestError` because the request is
    well-formed -- a caller-controlled retry after backoff is legitimate.
    This adapter still never retries the billable create automatically.
    """


class NimbleServerError(NimbleAgentError):
    """The Nimble API failed server-side (5xx).

    Distinct from :class:`NimbleInvalidRequestError` because nothing about
    the request is known to be wrong. For a create, note the outcome may be
    ambiguous -- the run may or may not exist -- which is exactly why this
    adapter never auto-retries creates.
    """


class NimbleInvalidResponseError(NimbleAgentError, IntegrationInvalidResponseException):
    """The Nimble API returned an unexpected, malformed, or inconsistent response."""


class NimbleRunFailedError(NimbleAgentError):
    """A run reached a terminal ``failed`` or ``cancelled`` state."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        agent_id: str | None = None,
        status: str | None = None,
        ref_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, run_id=run_id, agent_id=agent_id)
        self.status = status
        self.ref_id = ref_id
        self.details = details or {}


class NimbleRunTimeoutError(NimbleAgentError):
    """The configured poll deadline elapsed before a terminal state."""


class NimbleRunOwnershipError(NimbleAgentError):
    """A run's returned ``web_search_agent_id`` did not match what was expected."""
