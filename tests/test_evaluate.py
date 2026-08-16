"""Tests for evaluation metrics and baseline lifecycle."""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.baseline_lifecycle import (
    BaselineUpdatePolicy,
    commissioning_length,
    simulate_seasonal_drift,
)
from pumpwatch.evaluate import (
    bootstrap_ci,
    classify_report,
    expected_calibration_error,
    friedman_nemenyi_allowed,
    mcnemar_exact,
    recall_at_fixed_far,
)
from pumpwatch.gateway.baselines import MajorityClassifier, fit_predict, make_logistic


def test_commissioning_days_scale_with_features():
    p10 = commissioning_length(10, runtime_hours_per_day=3.0)
    p40 = commissioning_length(40, runtime_hours_per_day=3.0)
    assert p40.calendar_days > p10.calendar_days
    assert p10.min_samples == int(np.ceil(10 * 10 * 1.5))


def test_baseline_update_policy():
    pol = BaselineUpdatePolicy(max_age_days=30, recompute_on_n_new_healthy=100)
    assert pol.should_update(31, 0)
    assert pol.should_update(1, 100)
    assert not pol.should_update(1, 10)
    assert pol.require_authenticated_channel


def test_drift_sim_runs():
    r = simulate_seasonal_drift(n_features=5, n_healthy_fit=100, n_days=30, samples_per_day=5)
    assert r.false_alarms >= 0
    assert len(r.d2) == 30 * 5


def test_mcnemar_and_classify():
    y = np.array(["a", "a", "b", "b", "a", "b", "a", "b"])
    pred_a = np.array(["a", "a", "b", "a", "a", "b", "b", "b"])
    pred_b = np.array(["a", "b", "b", "b", "a", "b", "a", "b"])
    m = mcnemar_exact(y, pred_a, pred_b)
    assert "p_value" in m
    report = classify_report(y, pred_a)
    assert 0 <= report.macro_f1 <= 1


def test_recall_at_fixed_far():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9, 0.85, 0.6])
    r = recall_at_fixed_far(y, scores, max_far=0.25)
    assert r["far"] <= 0.25 + 1e-9
    assert 0 <= r["recall"] <= 1


def test_ece_and_bootstrap():
    y = np.array(["a", "b", "a", "b", "a", "b"])
    classes = np.array(["a", "b"])
    proba = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.1, 0.9],
        ]
    )
    ece = expected_calibration_error(y, proba, classes)
    assert 0 <= ece <= 1
    ci = bootstrap_ci(np.array([0.5, 0.6, 0.55, 0.7]), n_boot=200)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]


def test_friedman_guard():
    assert not friedman_nemenyi_allowed(3)
    assert friedman_nemenyi_allowed(5)


def test_majority_and_logistic_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))
    y = np.array(["h"] * 40 + ["f"] * 40)
    X[40:] += 1.5
    maj = fit_predict(MajorityClassifier(), X[:60], y[:60], X[60:], "majority")
    assert maj.y_pred.shape == (20,)
    lr = fit_predict(make_logistic(), X[:60], y[:60], X[60:], "logistic")
    assert lr.y_proba is not None
    assert lr.latency_predict_s >= 0
