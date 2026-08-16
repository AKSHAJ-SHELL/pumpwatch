"""Tests for confound audit and leakage-ladder splits."""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.audit import ConfoundError, assert_not_confounded, audit_confound
from pumpwatch.splits import (
    NORMALIZATION_STRATEGIES,
    SplitLevel,
    normalize_features,
    split_lomo,
    split_random_window,
)


def test_confound_detects_class_source_separation():
    # Dry-run only on ownrig; bearing only on twente — classic confound
    y = ["dry_run"] * 20 + ["bearing_outer"] * 20 + ["healthy"] * 20
    machines = ["own_1"] * 20 + ["twente_A"] * 20 + ["own_1"] * 10 + ["twente_A"] * 10
    sources = ["ownrig"] * 20 + ["twente"] * 20 + ["ownrig"] * 10 + ["twente"] * 10
    X = np.random.randn(60, 5)
    report = audit_confound(y, machines, sources, X=X)
    assert report.confounded
    with pytest.raises(ConfoundError):
        assert_not_confounded(report)


def test_within_source_machine_probe_is_warning_not_hard_fail():
    # Two machines, same source, classes on both — probe may fire as warning
    rng = np.random.default_rng(0)
    y = (["healthy"] * 20 + ["cavitation"] * 20) * 2
    machines = ["A"] * 40 + ["B"] * 40
    sources = ["twente"] * 80
    # Make features encode machine via a large offset
    X = rng.normal(size=(80, 5))
    X[40:] += 5.0
    report = audit_confound(y, machines, sources, X=X)
    assert not report.confounded
    assert report.feature_encodes_machine
    assert report.warnings
    assert_not_confounded(report)  # must not raise


def test_clean_within_source_not_confounded():
    y = ["healthy"] * 15 + ["cavitation"] * 15 + ["healthy"] * 15 + ["cavitation"] * 15
    machines = ["A"] * 30 + ["B"] * 30
    sources = ["twente"] * 60
    report = audit_confound(y, machines, sources, X=None, nmi_threshold=0.95)
    assert not any("only one machine" in r for r in report.reasons)


def test_lomo_holds_out_each_machine():
    machines = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    result = split_lomo(machines)
    assert result.level == SplitLevel.LEAVE_ONE_MACHINE_OUT
    assert result.verdict == "thesis_test"
    assert len(result.folds) == 3
    held = {f.held_out for f in result.folds}
    assert held == {"A", "B", "C"}
    for f in result.folds:
        assert f.held_out not in set(np.asarray(machines)[f.context_idx].tolist())


def test_random_window_marked_invalid():
    r = split_random_window(100, seed=0)
    assert r.verdict == "INVALID"


def _two_machine_problem(seed: int = 0):
    """Machine B lives on a wildly different scale from A — the LOMO failure mode."""
    rng = np.random.default_rng(seed)
    Xa = rng.normal(10.0, 1.0, size=(20, 3))
    Xb = rng.normal(1000.0, 100.0, size=(20, 3))
    X = np.vstack([Xa, Xb])
    machines = ["A"] * 20 + ["B"] * 20
    return X, machines


@pytest.mark.parametrize("strategy", NORMALIZATION_STRATEGIES)
def test_held_out_machine_is_always_normalized(strategy):
    """Regression: the held-out machine must never be left in raw units.

    The original implementation skipped any machine with no training rows, which
    under LOMO is exactly the machine being tested. Training rows were z-scored and
    test rows were not, so every model collapsed to chance.
    """
    X, machines = _two_machine_problem()
    fold = split_lomo(machines).folds[0]
    held = fold.held_out
    Xn = normalize_features(X, machines, fold.train_idx, strategy=strategy)

    test_rows = Xn[fold.test_idx]
    train_rows = Xn[fold.train_idx]
    assert np.isfinite(Xn).all()
    # The defect signature: test rows still carrying the raw scale.
    assert np.abs(test_rows).max() < 50.0, (
        f"held-out machine {held} left near-raw under {strategy}"
    )
    # Train and test must end up on a comparable scale, or the model sees garbage.
    assert np.abs(test_rows.std() - train_rows.std()) < 5.0


def test_unsupervised_per_machine_centres_every_machine():
    X, machines = _two_machine_problem()
    fold = split_lomo(machines).folds[0]
    Xn = normalize_features(
        X, machines, fold.train_idx, strategy="unsupervised_per_machine"
    )
    # Each machine standardised by its own statistics → both centred near zero.
    assert Xn[:20].mean() == pytest.approx(0.0, abs=1e-6)
    assert Xn[20:].mean() == pytest.approx(0.0, abs=1e-6)


def test_train_pooled_uses_no_test_statistics():
    """train_pooled must be invariant to the held-out machine's values."""
    X, machines = _two_machine_problem()
    fold = split_lomo(machines).folds[0]
    Xn_a = normalize_features(X, machines, fold.train_idx, strategy="train_pooled")

    X2 = X.copy()
    X2[fold.test_idx] *= 3.7  # perturb only the held-out machine
    Xn_b = normalize_features(X2, machines, fold.train_idx, strategy="train_pooled")

    # Training rows unchanged: the scaler saw nothing of the test machine.
    assert np.allclose(Xn_a[fold.train_idx], Xn_b[fold.train_idx])


def test_constant_feature_does_not_explode():
    X = np.hstack([np.random.default_rng(0).normal(size=(20, 2)), np.ones((20, 1))])
    machines = ["A"] * 10 + ["B"] * 10
    fold = split_lomo(machines).folds[0]
    for strategy in NORMALIZATION_STRATEGIES:
        Xn = normalize_features(X, machines, fold.train_idx, strategy=strategy)
        assert np.isfinite(Xn).all()
        assert np.abs(Xn[:, 2]).max() < 1e3


def test_unknown_strategy_rejected():
    X, machines = _two_machine_problem()
    fold = split_lomo(machines).folds[0]
    with pytest.raises(ValueError, match="unknown normalisation strategy"):
        normalize_features(X, machines, fold.train_idx, strategy="global")


def test_collapse_map_covers_every_twente_family():
    """Real Twente data has 15 fault families against TabPFN's hard cap of 10."""
    from pumpwatch.datasets.twente import (
        TWENTE_FAULT_FAMILIES,
        TABPFN_MAX_CLASSES,
        collapse_labels,
    )

    assert len(TWENTE_FAULT_FAMILIES) > TABPFN_MAX_CLASSES, "cap would not bind"
    collapsed = collapse_labels(TWENTE_FAULT_FAMILIES)
    assert len(set(collapsed)) <= TABPFN_MAX_CLASSES


def test_collapse_refuses_unknown_labels():
    """Silently dropping a class changes what the reported accuracy means."""
    from pumpwatch.datasets.twente import collapse_labels

    with pytest.raises(ValueError, match="no collapse rule"):
        collapse_labels(["healthy", "some_new_fault"])


def test_collapse_refuses_to_exceed_the_cap():
    from pumpwatch.datasets.twente import collapse_labels

    identity = {f"c{i}": f"c{i}" for i in range(12)}
    with pytest.raises(ValueError, match="above the .* cap"):
        collapse_labels(list(identity), mapping=identity, max_classes=10)
