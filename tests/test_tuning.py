"""Tests for nested hyperparameter search.

The safety property under test is that tuning never sees the outer fold's held-out
machine. If it does, the selected hyperparameters are chosen with knowledge of the
test set and every tuned baseline score is silently inflated — which is exactly the
class of defect this codebase has spent its time removing, so it is asserted here
rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.splits import split_lomo
from pumpwatch.tuning import (
    DEFAULT_GRIDS,
    TuningLeakError,
    _grid_points,
    inner_machine_folds,
    tune_model,
    tuned_factory,
)


def _dataset(n_machines=5, n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    X, y, machines = [], [], []
    for m in range(n_machines):
        for i in range(n_per):
            cls = i % 2
            X.append(rng.normal(3.0 * cls + 0.5 * m, 1.0, size=4))
            y.append(f"class{cls}")
            machines.append(f"m{m}")
    return np.array(X), np.array(y), machines


class _Dummy:
    """Minimal sklearn-ish model whose score depends on a hyperparameter."""

    def __init__(self, shift=0.0):
        self.shift = shift
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._means = {c: X[y == c].mean(axis=0) for c in self.classes_}
        return self

    def predict(self, X):
        keys = list(self._means)
        d = np.stack([np.linalg.norm(X - (self._means[c] + self.shift), axis=1) for c in keys])
        return np.array(keys)[np.argmin(d, axis=0)]


def test_grid_points_enumerates_the_product():
    pts = _grid_points({"a": [1, 2], "b": [10, 20, 30]})
    assert len(pts) == 6
    assert {"a": 1, "b": 10} in pts


def test_grid_points_handles_empty_grid():
    assert _grid_points({}) == [{}]


def test_inner_folds_stay_inside_the_training_pool():
    X, y, machines = _dataset()
    fold = split_lomo(machines).folds[0]
    folds = inner_machine_folds(np.asarray(machines), fold.train_idx)
    assert folds
    for tr, va in folds:
        assert set(tr).issubset(set(fold.train_idx.tolist()))
        assert set(va).issubset(set(fold.train_idx.tolist()))
        assert not set(tr) & set(va)


def test_inner_folds_are_grouped_by_machine():
    """A machine must never straddle the inner train/validation boundary."""
    X, y, machines = _dataset()
    m = np.asarray(machines)
    fold = split_lomo(machines).folds[0]
    for tr, va in inner_machine_folds(m, fold.train_idx):
        assert not set(m[tr].tolist()) & set(m[va].tolist())


def test_inner_folds_never_contain_the_held_out_machine():
    X, y, machines = _dataset()
    m = np.asarray(machines)
    for fold in split_lomo(machines).folds:
        for tr, va in inner_machine_folds(m, fold.train_idx):
            assert fold.held_out not in set(m[tr].tolist())
            assert fold.held_out not in set(m[va].tolist())


def test_tuning_raises_if_the_pool_contains_the_held_out_machine():
    """The safety assertion fires rather than silently tuning on the test machine."""
    X, y, machines = _dataset()
    all_idx = np.arange(len(y))  # deliberately includes every machine
    with pytest.raises(TuningLeakError, match="appears in the tuning pool"):
        tune_model(
            make_model=_Dummy,
            X=X, y=y, machines=machines, train_idx=all_idx,
            grid={"shift": [0.0, 5.0]},
            held_out_machine="m0",
        )


def test_tuning_selects_the_better_hyperparameter():
    X, y, machines = _dataset()
    fold = split_lomo(machines).folds[0]
    res = tune_model(
        make_model=_Dummy,
        X=X, y=y, machines=machines, train_idx=fold.train_idx,
        grid={"shift": [0.0, 50.0]},
        held_out_machine=fold.held_out,
    )
    # A huge shift wrecks the nearest-centroid rule, so 0.0 must win.
    assert res.best_params == {"shift": 0.0}
    assert res.n_candidates == 2
    assert res.n_inner_folds >= 2


def test_tuning_degrades_gracefully_with_too_few_machines():
    X, y, machines = _dataset(n_machines=2)
    fold = split_lomo(machines).folds[0]  # leaves exactly one training machine
    res = tune_model(
        make_model=_Dummy,
        X=X, y=y, machines=machines, train_idx=fold.train_idx,
        grid={"shift": [0.0, 5.0]},
        held_out_machine=fold.held_out,
    )
    assert res.n_inner_folds == 0
    assert "too few training machines" in res.note


def test_single_candidate_grid_skips_search():
    X, y, machines = _dataset()
    fold = split_lomo(machines).folds[0]
    res = tune_model(
        make_model=_Dummy,
        X=X, y=y, machines=machines, train_idx=fold.train_idx,
        grid={"shift": [0.0]},
        held_out_machine=fold.held_out,
    )
    assert "only one candidate" in res.note


def test_tuned_factory_produces_a_configured_model():
    X, y, machines = _dataset()
    fold = split_lomo(machines).folds[0]
    f = tuned_factory(
        _Dummy, {"shift": [0.0, 50.0]},
        X=X, y=y, machines=machines, train_idx=fold.train_idx,
        held_out_machine=fold.held_out,
    )
    model = f()
    assert model.shift == 0.0
    assert f.tuning_result.best_params == {"shift": 0.0}


def test_default_grids_are_small_enough_to_run():
    """A large grid multiplied over 11 outer folds stops being runnable."""
    for name, grid in DEFAULT_GRIDS.items():
        assert len(_grid_points(grid)) <= 12, f"{name} grid too large"


def test_default_grids_match_the_real_factory_signatures():
    """Guards the clf__C mistake: grid keys must be accepted by the factory.

    make_logistic forwards kwargs to LogisticRegression, not to the enclosing
    Pipeline, so Pipeline-style names raise at the first candidate.
    """
    from pumpwatch.gateway.baselines import make_logistic, make_lightgbm

    for params in _grid_points(DEFAULT_GRIDS["logistic"]):
        make_logistic(**params)
    for params in _grid_points(DEFAULT_GRIDS["lightgbm"]):
        make_lightgbm(**params)
