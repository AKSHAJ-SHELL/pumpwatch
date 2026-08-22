"""Smoke tests for the figure suite.

871 LOC with no tests. These do not check that a plot is *good* — they check it
renders at all from a minimal input, which is enough to catch the class of error
that has already occurred once here (`_save` receiving a str and raising
AttributeError on `.parent`), and the shape errors that come from a results dict
whose keys moved.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pumpwatch import figures as F


def test_save_accepts_str_and_path(tmp_path):
    """Regression: _save used to assume Path and crash on a plain string."""
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots()
    assert F._save(fig, str(tmp_path / "a.png")).exists()
    fig, _ = plt.subplots()
    assert F._save(fig, tmp_path / "b.png").exists()


def test_save_creates_missing_parent_directories(tmp_path):
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots()
    out = F._save(fig, tmp_path / "deep" / "nested" / "c.png")
    assert out.exists()


# --- figures needing no results file --------------------------------------


def test_physics_and_energy_figures_render(tmp_path):
    assert F.fig_cavitation_nonmonotonic(tmp_path / "a3.png").exists()
    assert F.fig_energy_battery_life(tmp_path / "e4.png").exists()
    assert F.fig_energy_breakdown(tmp_path / "e3.png").exists()
    assert F.fig_escalation_vs_battery(tmp_path / "c5.png", measured_rate=0.02).exists()


def test_vpf_and_dry_run_panels_render(tmp_path):
    assert F.fig_vpf_sidebands(tmp_path / "a6.png").exists()
    assert F.fig_dry_run_signature(tmp_path / "a7.png").exists()


def test_baseline_lifecycle_renders(tmp_path):
    """Orphan figure — kept tested so wiring it in cannot break silently."""
    assert F.fig_baseline_lifecycle(tmp_path / "bl.png").exists()


# --- figures driven by a results dict -------------------------------------


def test_leakage_ladder_renders(tmp_path):
    ladder = {
        "0_random_window": {"lightgbm": 1.0, "majority": 0.02},
        "4_lomo": {"lightgbm": 0.42, "majority": 0.02},
    }
    assert F.fig_leakage_ladder(tmp_path / "d1.png", ladder).exists()


def test_context_sweep_renders(tmp_path):
    sweep = [
        {"n_context": 50, "macro_f1": 0.64, "latency_predict_s": 0.4},
        {"n_context": 500, "macro_f1": 0.74, "latency_predict_s": 0.8},
    ]
    assert F.fig_context_sweep(tmp_path / "d4.png", sweep).exists()


def test_leakage_across_datasets_renders(tmp_path):
    entries = [
        {"dataset": "synthetic", "invalid": 1.0, "honest": 0.93, "honest_label": "LOMO"},
        {"dataset": "ESPset", "invalid": 0.79, "honest": 0.42, "honest_label": "LOMO"},
    ]
    assert F.fig_leakage_across_datasets(tmp_path / "d13.png", entries).exists()


def test_recall_at_alarm_budget_renders(tmp_path):
    budget = {
        "lightgbm": {"recall": 0.084, "max_far": 0.00093, "windows_per_month": 1080},
        "tabpfn": {"recall": 0.203, "max_far": 0.00093, "windows_per_month": 1080},
    }
    assert F.fig_recall_at_alarm_budget(tmp_path / "d12.png", budget).exists()


def test_detection_by_severity_renders(tmp_path):
    """Orphan figure — data already exists in the Twente results."""
    by_sev = {
        "1": {"n": 54, "detected_rate": 0.54, "correct_class_rate": 0.4},
        "3": {"n": 20, "detected_rate": 0.8, "correct_class_rate": 0.6},
    }
    assert F.fig_detection_by_severity(tmp_path / "sev.png", by_sev).exists()


def test_lomo_per_machine_renders(tmp_path):
    assert F.fig_lomo_per_machine(
        tmp_path / "d7.png", {"m0": 0.7, "m1": 0.6}, model_name="lightgbm"
    ).exists()


def test_accuracy_vs_latency_renders(tmp_path):
    pts = [
        {"model": "lightgbm", "macro_f1": 0.67, "latency_s": 0.001},
        {"model": "tabpfn", "macro_f1": 0.74, "latency_s": 0.8},
    ]
    assert F.fig_accuracy_vs_latency(tmp_path / "d2.png", pts).exists()


def test_calibration_tolerates_empty_reliability_bins(tmp_path):
    """A fold with no populated bins must not produce an all-NaN mean warning."""
    per_machine = {"m0": {"ece": 0.1, "reliability": {
        "bin_conf": [float("nan")] * 10, "bin_acc": [float("nan")] * 10,
    }}}
    assert F.fig_calibration(tmp_path / "d6.png", per_machine, label="x").exists()


def test_pca_class_vs_machine_renders(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    labels = np.array(["healthy"] * 20 + ["fault"] * 20)
    machines = np.array(["m0"] * 10 + ["m1"] * 10 + ["m2"] * 10 + ["m3"] * 10)
    assert F.fig_pca_class_vs_machine(tmp_path / "b6.png", X, labels, machines).exists()
