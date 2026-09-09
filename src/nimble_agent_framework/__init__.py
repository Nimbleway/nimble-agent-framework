"""Nimble Web Search Agent for Microsoft Agent Framework."""

from ._gating import GatedNimbleEffortError, GatePolicy
from .agent import CLIENT_SOURCE, NimbleWebSearchAgent
from .exceptions import (
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

__version__ = "0.1.0"

__all__ = [
    "CLIENT_SOURCE",
    "GatePolicy",
    "GatedNimbleEffortError",
    "NimbleAgentError",
    "NimbleAuthError",
    "NimbleInvalidRequestError",
    "NimbleInvalidResponseError",
    "NimbleRateLimitError",
    "NimbleRunFailedError",
    "NimbleRunOwnershipError",
    "NimbleRunTimeoutError",
    "NimbleServerError",
    "NimbleWebSearchAgent",
    "__version__",
]
