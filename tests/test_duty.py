"""Tests for the shared duty cycle, and for the split it exists to enforce.

The same number — 12 feature windows per runtime hour — used to be hardcoded in three
modules that had no way of knowing about each other. They agreed by coincidence. When
it turned out that holding the decision cadence at that rate was what capped end-to-end
recall at 0.086, changing it meant changing three places consistently or reporting an
alarm budget for one duty and a battery life for another.

These tests pin two things: that the modules take their rates from one place, and that
commissioning and decision cadences stay *different*, because conflating them is what
turns a three-day commissioning into a hundred-day one.
"""

from __future__ import annotations

import pytest

from pumpwatch import duty
from pumpwatch.baseline_lifecycle import commissioning_length
from pumpwatch.evaluate import (
    DEFAULT_RUNTIME_HOURS_PER_DAY,
    DEFAULT_WINDOWS_PER_RUNTIME_HOUR,
    far_for_alarms_per_month,
    windows_per_month,
)
from pumpwatch.node.energy import event_triggered_energy


def test_alarm_budget_uses_the_decision_cadence():
    assert DEFAULT_WINDOWS_PER_RUNTIME_HOUR == duty.DEFAULT_DECISION_WINDOWS_PER_RUNTIME_HOUR
    assert DEFAULT_RUNTIME_HOURS_PER_DAY == duty.DEFAULT_RUNTIME_HOURS_PER_DAY


def test_commissioning_uses_the_commissioning_cadence_not_the_decision_one():
    """The load-bearing assertion of this whole change.

    If commissioning ever picks up the decision rate, a seven-feature gate needs 105
    healthy windows at one per day and a node becomes usable after 106 days instead of
    three. Nothing else in the test suite would notice.
    """
    plan = commissioning_length(7)
    assert plan.samples_per_runtime_hour == (
        duty.DEFAULT_COMMISSIONING_WINDOWS_PER_RUNTIME_HOUR
    )
    assert plan.calendar_days < 5, (
        f"commissioning is {plan.calendar_days:.0f} days — the commissioning and "
        f"decision cadences have been conflated again"
    )


def test_the_two_cadences_are_actually_different():
    """A regression here means the split has been undone, not that a value changed."""
    assert (
        duty.DEFAULT_COMMISSIONING_WINDOWS_PER_RUNTIME_HOUR
        > duty.DEFAULT_DECISION_WINDOWS_PER_RUNTIME_HOUR
    )


def test_energy_model_uses_the_decision_cadence():
    """Every feature window the gate may escalate is one potential gateway decision."""
    e = event_triggered_energy(escalation_rate=0.059)
    expected = duty.DEFAULT_DUTY.decisions_per_day * 0.059
    assert e.transmissions_per_day == pytest.approx(expected, rel=1e-6)


def test_slower_cadence_does_not_cost_battery():
    """CUSUM sensing dominates, so cadence is close to free. Guards the claim."""
    fast = event_triggered_energy(feature_compute_per_runtime_hour=12.0, escalation_rate=0.059)
    slow = event_triggered_energy(escalation_rate=0.059)
    assert slow.battery_years >= fast.battery_years
    assert slow.transmissions_per_day < fast.transmissions_per_day


def test_default_duty_is_one_decision_per_runtime_day():
    d = duty.DEFAULT_DUTY
    assert d.decisions_per_day == pytest.approx(1.0)
    assert d.decisions_per_month == pytest.approx(30.0)
    assert d.far_for_alarms_per_month(1.0) == pytest.approx(1 / 30)


def test_far_matches_the_module_level_helper():
    """evaluate's helper and the dataclass must not drift apart either."""
    assert far_for_alarms_per_month(1.0) == pytest.approx(
        duty.DEFAULT_DUTY.far_for_alarms_per_month(1.0)
    )
    assert windows_per_month() == pytest.approx(duty.DEFAULT_DUTY.decisions_per_month)


def test_duty_for_decisions_per_month_leaves_commissioning_alone():
    """Sweeping the operational cadence must never lengthen commissioning."""
    for n in (1080, 90, 30, 12):
        d = duty.duty_for_decisions_per_month(n)
        assert d.decisions_per_month == pytest.approx(n)
        assert d.commissioning_windows_per_runtime_hour == (
            duty.DEFAULT_COMMISSIONING_WINDOWS_PER_RUNTIME_HOUR
        )


def test_alarm_promise_is_invariant_under_cadence():
    """The operator-facing promise does not move; only the per-decision rate does.

    This is the answer to the obvious objection that a slower cadence is metric-gaming:
    one tolerated alarm per pump-month buys exactly one tolerated alarm per pump-month
    at every cadence. What changes is how specific each decision must be.
    """
    for n in (1080, 30, 12):
        d = duty.duty_for_decisions_per_month(n)
        assert d.far_for_alarms_per_month(1.0) * d.decisions_per_month == pytest.approx(1.0)


def test_rejects_a_nonsensical_duty():
    with pytest.raises(ValueError):
        duty.DutyCycle(decision_windows_per_runtime_hour=0.0)
    with pytest.raises(ValueError):
        duty.duty_for_decisions_per_month(0)
