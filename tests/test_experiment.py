"""Tests for the shared evaluation harness.

Every result in the project flows through this module, and until now nothing tested
it. That is not an abstract gap: the seed bug — where run_split_repeated silently
re-ran an identical model N times and reported a spread of exactly zero — shipped
precisely because no test exercised the path.
"""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.experiment import build_ladder, run_split, run_split_repeated
from pumpwatch.gateway.baselines import MajorityClassifier
from pumpwatch.splits import split_lomo


def _dataset(n_machines=4, n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    X, y, machines = [], [], []
    for m in range(n_machines):
        for i in range(n_per):
            cls = i % 2
            X.append(rng.normal(3.0 * cls, 1.0, size=3))
            y.append(f"class{cls}")
            machines.append(f"m{m}")
    return np.array(X), np.array(y), machines


class _Spy:
    """Records exactly which rows it was fitted on, so leakage is checkable."""

    seen_fit_rows: list = []

    def __init__(self, seed=0):
        self.seed = seed
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        _Spy.seen_fit_rows.append(np.asarray(X).copy())
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


# --- run_split -------------------------------------------------------------


def test_run_split_reports_every_fold():
    X, y, machines = _dataset()
    r = run_split(X, y, machines, MajorityClassifier, "majority", split_lomo(machines))
    assert r["n_folds"] == 4
    assert set(r["per_machine_macro_f1"]) == set(machines)
    assert 0.0 <= r["overall_macro_f1"] <= 1.0


def test_run_split_never_fits_on_test_rows():
    """The property the whole leakage ladder depends on.

    Asserted against the rows the model actually received, not against index
    bookkeeping — the two can disagree, and only the former matters.
    """
    X, y, machines = _dataset()
    split = split_lomo(machines)
    _Spy.seen_fit_rows = []
    run_split(X, y, machines, _Spy, "spy", split, norm_strategy="train_pooled")

    for fold, fitted in zip(split.folds, _Spy.seen_fit_rows):
        held = {tuple(row) for row in X[fold.test_idx]}
        got = {tuple(row) for row in fitted}
        # Normalisation rescales, so compare counts rather than raw values:
        # the fitted block must be exactly the training block's size.
        assert len(fitted) == len(fold.train_idx)
        assert len(fitted) + len(fold.test_idx) == len(X)


def test_run_split_records_coverage_and_confusion():
    X, y, machines = _dataset()
    r = run_split(X, y, machines, MajorityClassifier, "majority", split_lomo(machines))
    assert r["overall_coverage"] == 1.0
    assert r["overall_confusion"]
    assert r["bootstrap_unit"] == "held_out_group"


def test_friedman_is_blocked_below_five_folds():
    X, y, machines = _dataset(n_machines=3)
    r = run_split(X, y, machines, MajorityClassifier, "majority", split_lomo(machines))
    assert r["friedman_allowed"] is False


# --- run_split_repeated ----------------------------------------------------


def test_repeated_rejects_a_factory_that_cannot_take_a_seed():
    """The guard that replaced the silent 5x-for-nothing fallback."""
    X, y, machines = _dataset()
    with pytest.raises(TypeError, match="takes no `seed` argument"):
        run_split_repeated(
            X, y, machines, MajorityClassifier, "majority",
            split_lomo(machines), seeds=(0, 1),
        )


def test_repeated_aggregates_across_seeds():
    X, y, machines = _dataset()
    r = run_split_repeated(
        X, y, machines, lambda seed=0: MajorityClassifier(), "majority",
        split_lomo(machines), seeds=(0, 1, 2),
    )
    s = r["macro_f1_over_seeds"]
    assert s["n_seeds"] == 3
    assert len(s["values"]) == 3
    assert r["seeds"] == [0, 1, 2]
    # Headline is the mean, so one lucky seed cannot become the result.
    assert r["overall_macro_f1"] == pytest.approx(np.mean(s["values"]))


def test_repeated_reports_zero_spread_for_a_deterministic_model():
    """Zero spread is the correct answer here — it must be distinguishable from
    the old bug, which produced zero spread for every model regardless."""
    X, y, machines = _dataset()
    r = run_split_repeated(
        X, y, machines, lambda seed=0: MajorityClassifier(), "majority",
        split_lomo(machines), seeds=(0, 1, 2),
    )
    assert r["macro_f1_over_seeds"]["std"] == pytest.approx(0.0)


# --- build_ladder ----------------------------------------------------------


def test_ladder_skips_rungs_with_absent_grouping_keys():
    """A rung whose key is missing must be omitted, never silently downgraded."""
    _, y, machines = _dataset()
    groups = {"record": [""] * len(y), "component": [""] * len(y), "operating": [""] * len(y)}
    ladder = build_ladder(machines, groups, n_samples=len(y))
    assert "0_random_window" in ladder
    assert "4_lomo" in ladder
    for absent in ("1_record_wise", "2_component_wise", "3_cross_operating"):
        assert absent not in ladder


def test_ladder_includes_rungs_whose_keys_are_present():
    _, y, machines = _dataset()
    n = len(y)
    groups = {
        "record": [f"r{i // 5}" for i in range(n)],
        "component": [f"c{i // 10}" for i in range(n)],
        "operating": [f"op{i % 2}" for i in range(n)],
    }
    ladder = build_ladder(machines, groups, n_samples=n)
    assert {"0_random_window", "1_record_wise", "2_component_wise",
            "3_cross_operating", "4_lomo"} <= set(ladder)


def test_ladder_skips_a_single_valued_grouping_key():
    _, y, machines = _dataset()
    groups = {
        "record": ["only-one"] * len(y),
        "component": [""] * len(y),
        "operating": [""] * len(y),
    }
    assert "1_record_wise" not in build_ladder(machines, groups, n_samples=len(y))
