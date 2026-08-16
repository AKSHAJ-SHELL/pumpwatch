"""Tests for confound audit and leakage-ladder splits."""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.audit import ConfoundError, assert_not_confounded, audit_confound
from pumpwatch.splits import (
    SplitLevel,
    normalize_per_machine,
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


def test_normalize_per_machine_no_leakage_fit():
    X = np.vstack([np.ones((20, 3)) * 10, np.ones((20, 3)) * 100])
    machines = ["A"] * 20 + ["B"] * 20
    train = np.arange(10)  # only first half of A — B has no train samples
    # B has no training samples: those rows stay unnormalised (skip)
    Xn = normalize_per_machine(X, machines, train)
    # A's train mean=10, so A's values → 0
    assert Xn[:20].mean() == pytest.approx(0.0, abs=1e-6)
