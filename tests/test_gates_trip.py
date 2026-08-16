"""Tests for gates and dry-run trip path."""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.node.gates import CUSUM1D, EWMAGate, MahalanobisGate
from pumpwatch.node.trip import (
    TripConfig,
    evaluate_trip_path,
    select_operating_point,
    sweep_trip_operating_points,
)
from pumpwatch.physics import DryRunCurrentParams, dry_run_current


def test_cusum_detects_down_shift():
    rng = np.random.default_rng(0)
    healthy = rng.normal(10.0, 0.2, size=200)
    cusum = CUSUM1D(k=0.5, h=4.0, direction="down").fit(healthy)
    t = np.linspace(0, 5, 100)
    i = dry_run_current(t, 1.0, DryRunCurrentParams(noise_std_fraction=0.01), "dry_run", rng)
    detected = False
    for x in i:
        hit, _ = cusum.update(float(x))
        if hit:
            detected = True
            break
    assert detected


def test_ewma_fit_and_update():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(100, 5))
    gate = EWMAGate(lam=0.3, n_sigma=3.0).fit(X)
    hit, score = gate.update(np.zeros(5))
    assert score.shape == (5,)
    assert isinstance(hit, bool)


def test_mahalanobis_rejects_small_n():
    X = np.random.randn(50, 10)  # n=50, p=10 → n < 10p
    with pytest.raises(ValueError, match="10p"):
        MahalanobisGate().fit(X)


def test_mahalanobis_distance_healthy_vs_outlier():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(300, 5))
    gate = MahalanobisGate(alpha=0.01).fit(X)
    d_ok = gate.distance(X.mean(axis=0))
    d_bad = gate.distance(X.mean(axis=0) + 5.0)
    assert d_ok < gate.threshold
    assert d_bad > gate.threshold
    exported = gate.export_baseline()
    assert "L" in exported and "mu" in exported


def test_trip_path_detects_dry_run_with_low_healthy_false_trips():
    result = evaluate_trip_path(n_trials=30, seed=0, duration_s=6.0)
    assert result.dry_run_detection_rate >= 0.8
    assert result.healthy_false_trip_rate <= 0.05
    # Seal destruction is sub-60 s (DESIGN §0.2); detection must land well inside it.
    assert result.dry_run_median_delay_s < 10.0


def test_trip_path_rejects_closed_valve_confuser():
    """The confuser rate is the whole justification for this path existing.

    DESIGN §0.2 says the trip path must earn its place partly through false-trip
    analysis against valve-throttle confusers. The previous assertion here was
    `0.0 <= rate <= 1.0`, a tautology, and it masked a shipped configuration that
    tripped on 100% of closed-valve events.
    """
    result = evaluate_trip_path(n_trials=30, seed=0, duration_s=6.0)
    assert result.closed_valve_false_trip_rate <= 0.05


def test_or_rule_is_what_broke_the_confuser_rejection():
    """Pin the defect so it cannot silently return.

    ORing the depth check with CUSUM makes the absolute floor decorative: CUSUM
    fires on any load loss, closed valve included.
    """
    ored = evaluate_trip_path(
        n_trials=30,
        seed=0,
        duration_s=6.0,
        trip_config=TripConfig(cusum_h=3.5, persistence_n=2, require_floor=False),
    )
    anded = evaluate_trip_path(n_trials=30, seed=0, duration_s=6.0)
    assert ored.closed_valve_false_trip_rate > 0.5
    assert anded.closed_valve_false_trip_rate < ored.closed_valve_false_trip_rate


def test_operating_point_selection_meets_its_budget():
    points = sweep_trip_operating_points(
        n_trials=12,
        seed=0,
        floor_fractions=[0.45, 0.55, 0.70],
        persistence_values=[2, 5],
        cusum_h_values=[3.5],
    )
    assert len(points) == 6
    chosen = select_operating_point(points, max_false_trip_rate=0.05)
    assert chosen is not None
    assert chosen["closed_valve_false_trip_rate"] <= 0.05
    assert chosen["healthy_false_trip_rate"] <= 0.05


def test_no_operating_point_when_budget_impossible():
    points = sweep_trip_operating_points(
        n_trials=8, seed=0,
        floor_fractions=[0.55], persistence_values=[2], cusum_h_values=[3.5],
    )
    assert select_operating_point(points, max_delay_s=1e-6) is None
