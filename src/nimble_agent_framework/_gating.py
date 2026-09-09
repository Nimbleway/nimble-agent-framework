"""Positive, auditable handling for unavailable Agent API V2 effort tiers.

``max`` is visible in the published ``effort`` schema but is not yet a
generally available tier.
This module keeps it *selectable* (never silently dropped) while guaranteeing
it is never sent to the API as if it were generally available: by default a
request for ``max`` raises with an explanation of what would ship instead and
how to opt in; ``gate_policy="degrade"`` opts into automatic substitution.
"""

from __future__ import annotations

from typing import Literal

GatePolicy = Literal["reject", "degrade"]

#: Tiers the Agent API V2 service treats as generally available today.
GENERALLY_AVAILABLE_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "x-high")

#: What a caller may select. ``max`` is included so intent can be expressed,
#: but it is resolved through :func:`resolve_effort` before anything billable
#: happens and is never sent as-is.
SELECTABLE_EFFORTS: tuple[str, ...] = (*GENERALLY_AVAILABLE_EFFORTS, "max")

_GATED_VALUE = "max"
_CLOSEST_AVAILABLE = "x-high"


class GatedNimbleEffortError(ValueError):
    """``effort="max"`` was requested under the (default) reject policy."""

    def __init__(self, *, policy: GatePolicy) -> None:
        self.requested = _GATED_VALUE
        self.policy = policy
        self.closest_available = _CLOSEST_AVAILABLE
        message = (
            "effort='max' is not generally available and is not sent to Nimble. "
            f"Pass effort='{_CLOSEST_AVAILABLE}' to proceed now, or construct "
            "NimbleWebSearchAgent with gate_policy='degrade' to opt into "
            f"automatic substitution of '{_CLOSEST_AVAILABLE}'."
        )
        super().__init__(message)


def resolve_effort(effort: str | None, *, gate_policy: GatePolicy = "reject") -> str | None:
    """Validate and resolve a per-run effort override.

    ``None`` (unset) always stays unset, so the agent/template server-side
    default applies -- effort is never silently raised. A generally
    available tier passes through unchanged. ``"max"`` is resolved per
    ``gate_policy``: ``"reject"`` (default) raises :class:`GatedNimbleEffortError`;
    ``"degrade"`` substitutes the closest generally available tier and never
    raises.
    """

    if effort is None:
        return None
    if effort in GENERALLY_AVAILABLE_EFFORTS:
        return effort
    if effort != _GATED_VALUE:
        raise ValueError(
            f"effort={effort!r} is not a supported Agent API V2 tier "
            f"(expected one of {SELECTABLE_EFFORTS!r})"
        )
    if gate_policy == "degrade":
        return _CLOSEST_AVAILABLE
    raise GatedNimbleEffortError(policy=gate_policy)
