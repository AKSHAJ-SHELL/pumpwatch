"""Tests for the shared per-machine gate in experiment.py.

These exist because the gate now produces a headline result — the escalation rate and
recall ceiling quoted for C1 on eleven in-service pumps — and until now it had no test
at all. Its two failure modes are both silent: a gate that escalates nothing looks
exactly like a gate that was never run, and an under-conditioned covariance produces
numbers that are arithmetically fine and meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.experiment import run_gate_per_machine, summarise_gate
from pumpwatch.node.gates import GATE_FEATURE_SETS

ORDER_FEATURES = GATE_FEATURE_SETS["order_spectrum"]


def _synthetic_population(
    n_machines: int = 3,
    n_healthy: int = 200,
    n_faulty: int = 40,
    separation: float = 6.0,
    seed: int = 0,
):
    """Machines whose faulty windows sit well away from their own healthy baseline.

    Each machine gets its own offset, so a gate fitted on the pooled population would
    behave differently from one fitted per machine — which is the property under test.
    """
    rng = np.random.default_rng(seed)
    p = len(ORDER_FEATURES)
    X, y, machines = [], [], []
    for m in range(n_machines):
        offset = 10.0 * m
        X.append(rng.normal(offset, 1.0, (n_healthy, p)))
        y += ["healthy"] * n_healthy
        X.append(rng.normal(offset + separation, 1.0, (n_faulty, p)))
        y += ["bearing"] * n_faulty
        machines += [f"pump_{m}"] * (n_healthy + n_faulty)
    return np.vstack(X), np.array(y), machines, list(ORDER_FEATURES)


def test_gate_escalates_faults_more_than_healthy():
    X, y, machines, names = _synthetic_population()
    res = run_gate_per_machine(X, y, machines, names, verbose=False)

    assert set(res) == {"pump_0", "pump_1", "pump_2"}
    for machine, stats in res.items():
        assert stats["escalation_rate_faulty"] > stats["escalation_rate_healthy"], machine
        # Separated by 6 sigma: the gate should catch essentially all of it.
        assert stats["escalation_rate_faulty"] > 0.9, machine


def test_gate_is_deterministic_given_a_seed():
    """The commissioning split is drawn at random, so the seed must control it."""
    args = _synthetic_population()
    a = run_gate_per_machine(*args, seed=7, verbose=False)
    b = run_gate_per_machine(*args, seed=7, verbose=False)
    c = run_gate_per_machine(*args, seed=8, verbose=False)

    assert a["pump_0"]["escalation_rate_field"] == b["pump_0"]["escalation_rate_field"]
    # Different seed draws a different commissioning half. Not asserting inequality —
    # on well-separated data it may legitimately coincide — only that it still runs.
    assert set(c) == set(a)


def test_commissioning_shortfall_is_reported_not_hidden():
    """A gate fitted on too few healthy windows must be flagged, not silently trusted."""
    # 7 gate features -> needs ceil(10*7*1.5) = 105 commissioning rows. Half of the
    # healthy windows commission the node, so 60 healthy gives 30: well short.
    X, y, machines, names = _synthetic_population(n_machines=1, n_healthy=60)
    res = run_gate_per_machine(X, y, machines, names, verbose=False)

    if res:  # Mahalanobis may refuse outright, which is also acceptable behaviour.
        stats = res["pump_0"]
        assert stats["commissioning_adequate"] is False
        assert stats["n_commissioning"] < stats["commissioning_required"]


def test_adequately_commissioned_when_healthy_history_is_long():
    X, y, machines, names = _synthetic_population(n_machines=1, n_healthy=400)
    res = run_gate_per_machine(X, y, machines, names, verbose=False)
    assert res["pump_0"]["commissioning_adequate"] is True


def test_machine_with_no_healthy_history_is_skipped():
    """A pump commissioned with no healthy baseline cannot be gated at all."""
    X, y, machines, names = _synthetic_population(n_machines=2)
    y = np.array(["bearing" if m == "pump_1" else lbl for lbl, m in zip(y, machines)])
    res = run_gate_per_machine(X, y, machines, names, verbose=False)
    assert "pump_1" not in res
    assert "pump_0" in res


def test_unknown_feature_schema_skips_rather_than_raising():
    """The gate is one step of a larger run, so an ungateable schema must not abort it.

    It must still be visible: returning {} is what summarise_gate then reports as
    absent, rather than as a gate that escalated nothing.
    """
    X, y, machines, _ = _synthetic_population()
    bogus = [f"not_a_gate_feature_{i}" for i in range(X.shape[1])]
    assert run_gate_per_machine(X, y, machines, bogus, verbose=False) == {}


def test_summarise_gate_reports_ceiling_and_commissioning_count():
    X, y, machines, names = _synthetic_population()
    res = run_gate_per_machine(X, y, machines, names, verbose=False)
    summary = summarise_gate(res)

    assert summary["n_machines"] == 3
    assert 0.0 <= summary["mean_field_escalation_rate"] <= 1.0
    assert summary["gate_recall_ceiling"] > 0.9
    assert summary["battery_years_at_field_rate"] > 0
    assert summary["uplinks_per_day_at_field_rate"] > 0
    assert summary["n_machines_adequately_commissioned"] <= summary["n_machines"]


def test_summarise_gate_on_no_results_is_empty_not_zero():
    """Distinguishing 'no gate ran' from 'the gate escalated nothing' is the point.

    A zero-filled summary would be indistinguishable from a perfectly quiet gate and
    would flow into the energy figures as a real measurement.
    """
    assert summarise_gate({}) == {}


def test_field_rate_is_dominated_by_healthy_false_escalation():
    """Battery life is driven by the field rate, and faults are ~1% of field traffic.

    This is the arithmetic the architecture claim rests on: at 1% prevalence the field
    escalation rate must sit near the healthy rate, not near the test-set rate, which
    reflects how many faulty examples happened to be collected.
    """
    X, y, machines, names = _synthetic_population()
    res = run_gate_per_machine(
        X, y, machines, names, field_fault_prevalence=0.01, verbose=False
    )
    for machine, s in res.items():
        assert s["escalation_rate_field"] == pytest.approx(
            0.99 * s["escalation_rate_healthy"] + 0.01 * s["escalation_rate_faulty"],
            abs=1e-9,
        ), machine
        # And therefore far below the test-set rate, where faults are over-represented.
        assert s["escalation_rate_field"] < s["escalation_rate_overall"], machine
