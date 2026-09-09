"""Effort gating: never silently raised, `max` never silently sent or dropped."""

from __future__ import annotations

import pytest

from nimble_agent_framework._gating import (
    GENERALLY_AVAILABLE_EFFORTS,
    GatedNimbleEffortError,
    resolve_effort,
)


def test_unset_effort_stays_unset() -> None:
    assert resolve_effort(None) is None


@pytest.mark.parametrize("effort", GENERALLY_AVAILABLE_EFFORTS)
def test_generally_available_effort_passes_through(effort: str) -> None:
    assert resolve_effort(effort) == effort
    assert resolve_effort(effort, gate_policy="degrade") == effort


def test_max_rejected_by_default() -> None:
    with pytest.raises(GatedNimbleEffortError) as excinfo:
        resolve_effort("max")
    assert excinfo.value.requested == "max"
    assert excinfo.value.closest_available == "x-high"
    assert "x-high" in str(excinfo.value)
    assert "gate_policy='degrade'" in str(excinfo.value)


def test_max_rejected_explicitly() -> None:
    with pytest.raises(GatedNimbleEffortError):
        resolve_effort("max", gate_policy="reject")


def test_max_degrades_to_x_high_when_opted_in() -> None:
    assert resolve_effort("max", gate_policy="degrade") == "x-high"


def test_unsupported_effort_value_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not a supported Agent API V2 tier"):
        resolve_effort("ultra")
