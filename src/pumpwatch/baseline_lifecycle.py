"""Baseline lifecycle: commissioning length, drift, update policy.

Mahalanobis needs n > 10p healthy samples. On an event-triggered node that
is calendar days, not a footnote. Seasonal suction-head drift moves μ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pumpwatch import duty
from pumpwatch.node.gates import MahalanobisGate


@dataclass
class CommissioningPlan:
    n_features: int
    min_samples: int
    samples_per_runtime_hour: float
    runtime_hours_per_day: float
    calendar_days: float
    notes: str


def commissioning_length(
    n_features: int,
    samples_per_runtime_hour: float = duty.DEFAULT_COMMISSIONING_WINDOWS_PER_RUNTIME_HOUR,
    runtime_hours_per_day: float = duty.DEFAULT_RUNTIME_HOURS_PER_DAY,
    safety_factor: float = 1.5,
) -> CommissioningPlan:
    """Days of healthy operation needed before Mahalanobis gate is well-conditioned.

    Defaults to the **commissioning** sampling rate, deliberately not the decision
    rate. Learning a baseline and making operational decisions are separate schedules
    (see ``pumpwatch.duty``): a node samples densely for the few days it takes to
    condition the covariance, then sparsely thereafter. Tying this to the decision
    cadence would make a slower operational cadence lengthen commissioning from about
    three days to over a hundred, which is the trap this split exists to avoid.
    """
    min_samples = int(np.ceil(10 * n_features * safety_factor))
    samples_per_day = samples_per_runtime_hour * runtime_hours_per_day
    if samples_per_day <= 0:
        raise ValueError("samples_per_day must be positive")
    days = min_samples / samples_per_day
    return CommissioningPlan(
        n_features=n_features,
        min_samples=min_samples,
        samples_per_runtime_hour=samples_per_runtime_hour,
        runtime_hours_per_day=runtime_hours_per_day,
        calendar_days=days,
        notes=(
            f"p={n_features} → need >{10 * n_features} samples (×{safety_factor} safety). "
            f"At {runtime_hours_per_day} h/day runtime this is {days:.1f} calendar days. "
            "Brownfield pumps may already be faulty — cold-start assumes health."
        ),
    )


@dataclass
class CommissioningProgress:
    n_features: int
    observed_running_windows: int
    required_windows: int
    commissioned: bool
    fraction: float
    note: str


def commissioning_progress(
    observed_running_windows: int,
    n_features: int,
    safety_factor: float = 1.5,
) -> CommissioningProgress:
    """Is this node commissioned yet, judged on windows actually observed?

    ``commissioning_length`` answers how long commissioning *should* take under a
    nominal duty. This answers whether it has in fact happened, which is a different
    question and the one that matters in the field: a pump that runs less than its
    nominal duty accumulates the baseline more slowly, and nothing was checking.

    The case that motivated it is a plant pump observed over three days that never
    accumulated more than 55 running windows against the 120 its gate needed. Without
    this check it was silently commissioned on half a baseline, and its escalation rate
    reported as though it meant something. "Not yet commissioned" is a state a
    deployment must be able to be in, and to report.
    """
    required = int(np.ceil(10 * n_features * safety_factor))
    observed = int(observed_running_windows)
    ok = observed >= required
    return CommissioningProgress(
        n_features=n_features,
        observed_running_windows=observed,
        required_windows=required,
        commissioned=ok,
        fraction=observed / required if required else 0.0,
        note=(
            f"{observed}/{required} running windows"
            + ("" if ok else " — NOT commissioned; any escalation rate from this "
                             "baseline is uninterpretable")
        ),
    )


@dataclass
class DriftSimResult:
    days: np.ndarray
    d2: np.ndarray
    false_alarms: int
    threshold: float


def simulate_seasonal_drift(
    n_features: int = 20,
    n_healthy_fit: int = 400,
    n_days: int = 180,
    samples_per_day: int = 20,
    drift_per_day: float = 0.01,
    seed: int = 0,
) -> DriftSimResult:
    """Simulate slow mean drift (seasonal head change) and count false alarms."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0, 1, size=(n_healthy_fit, n_features))
    gate = MahalanobisGate(alpha=0.01).fit(X0)

    days = []
    d2s = []
    false_alarms = 0
    for d in range(n_days):
        drift = drift_per_day * d
        X = rng.normal(drift, 1, size=(samples_per_day, n_features))
        for x in X:
            hit, d2 = gate.update(x)
            days.append(d)
            d2s.append(d2)
            if hit:
                false_alarms += 1
    return DriftSimResult(
        days=np.asarray(days),
        d2=np.asarray(d2s),
        false_alarms=false_alarms,
        threshold=gate.threshold,
    )


@dataclass
class BaselineUpdatePolicy:
    """When/how the gateway recomputes and ships a new Cholesky factor."""

    max_age_days: float = 30.0
    recompute_on_n_new_healthy: int = 100
    require_authenticated_channel: bool = True
    notes: str = (
        "Ship μ and L over authenticated LoRa (or out-of-band). "
        "Raw LoRa P2P without auth means a spoofed packet redefines 'normal'."
    )

    def should_update(self, age_days: float, n_new_healthy: int) -> bool:
        return age_days >= self.max_age_days or n_new_healthy >= self.recompute_on_n_new_healthy
