"""Baseline lifecycle: commissioning length, drift, update policy.

Mahalanobis needs n > 10p healthy samples. On an event-triggered node that
is calendar days, not a footnote. Seasonal suction-head drift moves μ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    samples_per_runtime_hour: float = 12.0,
    runtime_hours_per_day: float = 3.0,
    safety_factor: float = 1.5,
) -> CommissioningPlan:
    """Days of healthy operation needed before Mahalanobis gate is well-conditioned."""
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
