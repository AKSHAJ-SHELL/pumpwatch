"""Tests for run-state detection, and a regression guard for the bug it exists to fix.

A gate commissioned on a mostly-idle stretch of plant telemetry escalated 100 % of the
following running period, and that was written up as evidence that plant demand defeats
a commissioned baseline. It was evidence that the baseline described a stopped pump.
``test_idle_commissioning_reproduces_the_published_bug`` reproduces the failure so that
the fix cannot quietly regress and the mistake cannot be made a second time.
"""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.baseline_lifecycle import commissioning_progress
from pumpwatch.node.gates import fit_composite_gate
from pumpwatch.node.runstate import (
    RunState,
    RunStateConfig,
    RunStateDetector,
    config_from_healthy_load,
    running_mask,
)


def test_unknown_is_not_silently_running():
    """A missing load reading must not be reported as a running machine.

    Defaulting to RUNNING is how an idle baseline gets learned; defaulting to OFF is how
    a monitor goes quiet on a working pump. Neither is acceptable, so UNKNOWN exists.
    """
    d = RunStateDetector(RunStateConfig(on_threshold=10.0, off_threshold=5.0))
    assert d.update(None) == RunState.UNKNOWN
    assert d.update(float("nan")) == RunState.UNKNOWN


def test_unknown_does_not_change_the_committed_state():
    """Telemetry drops packets. The detector must track the pump, not the radio."""
    d = RunStateDetector(RunStateConfig(on_threshold=10.0, off_threshold=5.0, dwell_windows=1))
    d.update(20.0)
    assert d.state == RunState.RUNNING
    d.update(None)
    assert d.state == RunState.RUNNING


def test_hysteresis_stops_chatter_at_the_boundary():
    """A pump idling between the thresholds must hold state, not oscillate.

    Every transition resets the dwell counter, so a chattering detector never settles
    long enough to commission anything.
    """
    cfg = RunStateConfig(on_threshold=10.0, off_threshold=5.0, dwell_windows=1)
    d = RunStateDetector(cfg)
    d.update(20.0)
    for mid in (7.0, 6.0, 8.0, 9.0, 5.5):
        assert d.update(mid) == RunState.RUNNING, "band value should hold state"


def test_dwell_requires_sustained_evidence():
    cfg = RunStateConfig(on_threshold=10.0, off_threshold=5.0, dwell_windows=3)
    d = RunStateDetector(cfg)
    assert d.update(20.0) == RunState.OFF, "one sample is not enough"
    assert d.update(20.0) == RunState.OFF
    assert d.update(20.0) == RunState.RUNNING


def test_equal_thresholds_are_rejected():
    with pytest.raises(ValueError, match="hysteresis"):
        RunStateConfig(on_threshold=5.0, off_threshold=5.0)


def test_config_derived_from_a_duty_cycled_load():
    """Thresholds come from the data because units differ per deployment.

    A plain median of a duty-cycled machine sits in the idle floor, so the threshold is
    derived from the load *above* its own floor.
    """
    load = np.concatenate([np.full(90, 0.4), np.full(10, 40.0)])   # 90 % idle
    cfg = config_from_healthy_load(load)
    assert cfg.on_threshold > 1.0, "threshold must not land in the idle floor"
    assert cfg.off_threshold < cfg.on_threshold
    mask = running_mask(load, cfg)
    assert mask[:90].sum() == 0
    assert mask[90:].sum() >= 8


def test_a_machine_that_never_runs_is_reported_not_assumed():
    with pytest.raises(ValueError, match="never to run"):
        config_from_healthy_load(np.full(100, 0.4))


def _idle_and_running(seed: int = 0):
    """A 90 % idle commissioning block, then a running evaluation block.

    Mirrors the CIRA telemetry that produced the bug: the commissioning day was 89-92 %
    idle across the three pumps.
    """
    rng = np.random.default_rng(seed)
    n_feat = 4
    idle = rng.normal(0.5, 0.05, (180, n_feat))
    run_c = rng.normal(20.0, 1.0, (20, n_feat))
    commissioning = np.vstack([idle, run_c])
    load_c = np.concatenate([np.full(180, 0.5), np.full(20, 40.0)])
    evaluation = rng.normal(20.0, 1.0, (200, n_feat))
    load_e = np.full(200, 40.0)
    return commissioning, load_c, evaluation, load_e


def test_idle_commissioning_reproduces_the_published_bug():
    """Without run-state gating, an idle-commissioned gate escalates everything.

    This is the regression guard for the whole exercise. If this test ever stops
    failing-to-escalate-sanely, the bug is back.
    """
    Xc, _, Xe, _ = _idle_and_running()
    gate = fit_composite_gate(Xc, feature_names=[f"f{i}" for i in range(Xc.shape[1])])
    esc = np.mean([gate.update(x)["escalate"] for x in Xe])
    assert esc > 0.9, (
        f"expected the idle-commissioned gate to escalate nearly everything, got {esc:.2f}"
    )


def test_run_state_gating_fixes_it():
    """Commissioning on running windows only gives a usable baseline."""
    Xc, load_c, Xe, _ = _idle_and_running()
    cfg = config_from_healthy_load(load_c)
    mask = running_mask(load_c, cfg)
    gate = fit_composite_gate(Xc[mask], feature_names=[f"f{i}" for i in range(Xc.shape[1])])
    esc = np.mean([gate.update(x)["escalate"] for x in Xe])
    assert esc < 0.5, f"run-state-gated commissioning should not escalate everything: {esc:.2f}"


def test_gate_does_not_advance_state_while_off():
    """An idle stretch must not drag the recursive detectors' reference down.

    The EWMA and CUSUM are recursive. Feeding them off-windows is what turns a correct
    baseline into one that escalates the next running period wholesale, so the gate must
    return early rather than merely suppressing the alarm.
    """
    Xc, load_c, _, _ = _idle_and_running()
    mask = running_mask(load_c, config_from_healthy_load(load_c))
    names = [f"f{i}" for i in range(Xc.shape[1])]
    running_rows = Xc[mask]

    gated = fit_composite_gate(running_rows, feature_names=names)
    for _ in range(200):                       # a long idle stretch
        out = gated.update(np.full(Xc.shape[1], 0.5), run_state=RunState.OFF)
        assert out["escalate"] is False
        assert "not_running" in out["reasons"]

    fresh = fit_composite_gate(running_rows, feature_names=names)
    probe = running_rows.mean(axis=0)
    assert gated.update(probe)["escalate"] == fresh.update(probe)["escalate"], (
        "the idle stretch changed the gate's internal state despite being gated off"
    )


def test_default_call_is_unchanged():
    """No run state supplied must behave exactly as before — ESPset relies on it."""
    Xc, _, Xe, _ = _idle_and_running()
    names = [f"f{i}" for i in range(Xc.shape[1])]
    a = fit_composite_gate(Xc, feature_names=names)
    b = fit_composite_gate(Xc, feature_names=names)
    assert [a.update(x)["escalate"] for x in Xe] == [
        b.update(x, run_state=RunState.RUNNING)["escalate"] for x in Xe
    ]


def test_commissioning_progress_catches_the_uncommissionable_pump():
    """Pump C: 55 running windows against 120 required, silently used before."""
    p = commissioning_progress(observed_running_windows=55, n_features=8)
    assert p.required_windows == 120
    assert p.commissioned is False
    assert "NOT commissioned" in p.note
    assert commissioning_progress(120, 8).commissioned is True
