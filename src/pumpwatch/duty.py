"""The node's duty cycle — one definition, two deliberately different cadences.

This module exists because the same number, 12 feature windows per runtime hour, was
hardcoded independently in three places: the alarm-budget arithmetic
(``evaluate.DEFAULT_WINDOWS_PER_RUNTIME_HOUR``), the energy model
(``node.energy.event_triggered_energy``) and the commissioning-length calculation
(``baseline_lifecycle.commissioning_length``). Three copies agreeing by coincidence is
the drift pattern this project has repeatedly had to undo, and here it was actively
dangerous: changing the cadence in one place and not the others would report an alarm
budget for one duty and a battery life for another.

**The two cadences are not the same thing, and conflating them is the trap.**

``commissioning_windows_per_runtime_hour`` is how densely the node samples while it is
learning a new pump's healthy baseline. The Mahalanobis gate needs n > 10p healthy
windows before its covariance is usable at all, so this rate sets how many *days* a
node must observe before it can do anything. At 12/runtime-hour a seven-feature gate is
commissioned in about three days.

``decision_windows_per_runtime_hour`` is how often the node makes a classification
decision once it is running. This one sets the false-alarm arithmetic: a promise of one
tolerated alarm per pump-month divided by the number of decisions in that month gives
the per-decision false-alarm rate the classifier must hit.

Holding them equal at 12/runtime-hour is what capped end-to-end recall at 0.086. It
forces ~1080 decisions a month, so one tolerated alarm demands 99.907 % specificity per
decision. Dropping the decision rate to daily relaxes that to 96.7 % and takes recall
from 0.10 to 0.59 — but dropping the *commissioning* rate with it would push
commissioning from 3 days to 106.

There is no reason to tie them. Commissioning is a one-time calibration phase; the node
can sample densely for three days and sparsely thereafter. That is a schedule, not a
capability, and it is what these defaults encode.
"""

from __future__ import annotations

from dataclasses import dataclass

# Sampling rate while learning a pump's healthy baseline. Governs commissioning
# duration only.
DEFAULT_COMMISSIONING_WINDOWS_PER_RUNTIME_HOUR = 12.0

# Sampling rate for operational classification decisions. Governs the false-alarm
# budget only. One decision per runtime day at the default 3 h/day runtime.
DEFAULT_DECISION_WINDOWS_PER_RUNTIME_HOUR = 1.0 / 3.0

DEFAULT_RUNTIME_HOURS_PER_DAY = 3.0
DEFAULT_DAYS_PER_MONTH = 30.0
DEFAULT_ALARMS_PER_MONTH = 1.0


@dataclass(frozen=True)
class DutyCycle:
    """A node's sampling schedule, with commissioning and decision rates separate."""

    commissioning_windows_per_runtime_hour: float = (
        DEFAULT_COMMISSIONING_WINDOWS_PER_RUNTIME_HOUR
    )
    decision_windows_per_runtime_hour: float = DEFAULT_DECISION_WINDOWS_PER_RUNTIME_HOUR
    runtime_hours_per_day: float = DEFAULT_RUNTIME_HOURS_PER_DAY
    days_per_month: float = DEFAULT_DAYS_PER_MONTH

    def __post_init__(self) -> None:
        for name in (
            "commissioning_windows_per_runtime_hour",
            "decision_windows_per_runtime_hour",
            "runtime_hours_per_day",
            "days_per_month",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    @property
    def decisions_per_month(self) -> float:
        """Classification decisions in a pump-month. The alarm-budget denominator."""
        return (
            self.decision_windows_per_runtime_hour
            * self.runtime_hours_per_day
            * self.days_per_month
        )

    @property
    def decisions_per_day(self) -> float:
        return self.decision_windows_per_runtime_hour * self.runtime_hours_per_day

    @property
    def hours_between_decisions(self) -> float:
        """Detection latency added by the cadence, in wall-clock hours."""
        return 24.0 / self.decisions_per_day

    def far_for_alarms_per_month(
        self, alarms_per_month: float = DEFAULT_ALARMS_PER_MONTH
    ) -> float:
        """Per-decision false-alarm rate implied by an alarms-per-month promise.

        The promise is the operator-facing invariant and does not change with cadence.
        What changes is how many decisions that one tolerated alarm is spread across:
        fewer decisions means each may be less specific for the same promise.
        """
        return float(alarms_per_month) / self.decisions_per_month

    def describe(self) -> str:
        per_day = self.decisions_per_day
        if per_day >= 1:
            cadence = f"{per_day:.0f}/day"
        else:
            cadence = f"every {1 / per_day:.1f} days"
        return (
            f"{self.decisions_per_month:.0f} decisions/month ({cadence}), "
            f"FAR budget {self.far_for_alarms_per_month():.5f}; "
            f"commissioning at {self.commissioning_windows_per_runtime_hour:g}/runtime-hour"
        )


#: The schedule the system ships with. Named so results files can record which duty
#: produced them — a recall number without its cadence is not interpretable, exactly as
#: a cross-machine score without its normalisation strategy is not.
DEFAULT_DUTY = DutyCycle()


def duty_for_decisions_per_month(
    decisions_per_month: float, base: DutyCycle | None = None
) -> DutyCycle:
    """A duty cycle at a given decision rate, leaving commissioning untouched.

    The sweep in the paper varies only the decision cadence. Commissioning stays dense
    so that a slower operational cadence never lengthens the time before a node is
    usable.
    """
    base = base or DEFAULT_DUTY
    if decisions_per_month <= 0:
        raise ValueError("decisions_per_month must be positive")
    return DutyCycle(
        commissioning_windows_per_runtime_hour=base.commissioning_windows_per_runtime_hour,
        decision_windows_per_runtime_hour=(
            decisions_per_month / (base.runtime_hours_per_day * base.days_per_month)
        ),
        runtime_hours_per_day=base.runtime_hours_per_day,
        days_per_month=base.days_per_month,
    )
