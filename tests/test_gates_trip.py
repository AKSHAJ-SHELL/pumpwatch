"""Tests for gates and dry-run trip path."""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.node.gates import CUSUM1D, EWMAGate, MahalanobisGate
from pumpwatch.node.trip import TripConfig, evaluate_trip_path
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
    # Tuned for synthetic: catch dry-run; healthy false trips must stay rare.
    # Closed-valve false-trip rate is the figure of interest, not a hard fail.
    result = evaluate_trip_path(
        n_trials=30,
        seed=0,
        trip_config=TripConfig(cusum_h=5.0, persistence_n=5, absolute_floor_fraction=0.55),
        duration_s=6.0,
    )
    assert result.dry_run_detection_rate >= 0.8
    assert result.healthy_false_trip_rate <= 0.1
    assert result.dry_run_median_delay_s < 3.0
    assert 0.0 <= result.closed_valve_false_trip_rate <= 1.0
