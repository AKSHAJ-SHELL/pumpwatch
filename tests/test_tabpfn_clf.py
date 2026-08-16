"""Tests for the TabPFN v2 wrapper.

This module had no tests at all, and tabpfn was not installed, so none of it had
ever executed — including the version pin that the commercial-use argument rests on
and the guards that stop an oversized context reaching the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.gateway.tabpfn_clf import (
    ATTRIBUTION_NOTICE,
    COMMERCIAL_LICENCE_MARKER,
    AbstentionConfig,
    CachedTabPFN,
    TabPFNConfig,
    TabPFNVersionError,
    installed_tabpfn_licence,
    installed_tabpfn_version,
    tabpfn_available,
)

tabpfn_required = pytest.mark.skipif(
    not tabpfn_available(), reason="tabpfn not installed"
)


def _toy(n_per_class=30, n_features=6, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [rng.normal(3.0 * k, 1.0, (n_per_class, n_features)) for k in range(n_classes)]
    )
    y = np.array(sum(([f"class{k}"] * n_per_class for k in range(n_classes)), []))
    return X, y


def test_predict_requires_fit_context():
    clf = CachedTabPFN()
    with pytest.raises(RuntimeError, match="fit_context"):
        clf.predict(np.zeros((3, 4)))


def test_class_cap_is_enforced():
    """TabPFN's 10-class cap is architectural, not a tuning parameter.

    The Twente taxonomy lists 15 fault families, so this guard is the thing standing
    between real data and a confusing failure deep inside the model.
    """
    X, y = _toy(n_per_class=2, n_classes=11)
    clf = CachedTabPFN()
    with pytest.raises(ValueError, match="10-class cap"):
        clf.fit_context(X, y)


def test_row_cap_is_enforced_when_subsampling_is_off():
    """With subsampling disabled the architectural 10k cap must still bite."""
    clf = CachedTabPFN(config=TabPFNConfig(max_context_rows=None))
    X = np.zeros((10_001, 3))
    y = np.array(["a"] * 10_001)
    with pytest.raises(ValueError, match="10_000|10000"):
        clf.fit_context(X, y)


def test_oversized_context_is_subsampled_rather_than_rejected():
    """With subsampling on, a large context is handled instead of erroring."""
    rng = np.random.default_rng(0)
    clf = CachedTabPFN(config=TabPFNConfig(max_context_rows=500))
    X = rng.normal(size=(12_000, 3))
    y = np.array(["a", "b"] * 6_000)
    Xs, ys = clf.subsample_context(X, y)
    assert len(ys) == 500
    assert clf.context_subsampled_from == 12_000


def test_unpinned_package_is_refused_by_default(monkeypatch):
    """A non-commercial model must not silently produce publishable results.

    The previous wrapper warned and carried on, tagging results 'fallback' with
    nothing downstream refusing them.
    """
    monkeypatch.setattr(
        "pumpwatch.gateway.tabpfn_clf.installed_tabpfn_version", lambda: "8.3.0"
    )
    monkeypatch.setattr(
        "pumpwatch.gateway.tabpfn_clf.installed_tabpfn_licence", lambda: "Research only"
    )
    clf = CachedTabPFN()
    with pytest.raises(TabPFNVersionError, match="commercially usable"):
        clf._build_model()


def test_allow_unpinned_is_an_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(
        "pumpwatch.gateway.tabpfn_clf.installed_tabpfn_version", lambda: "8.3.0"
    )
    monkeypatch.setattr(
        "pumpwatch.gateway.tabpfn_clf.installed_tabpfn_licence", lambda: "Research only"
    )
    clf = CachedTabPFN(config=TabPFNConfig(allow_unpinned=True))
    if not tabpfn_available():
        pytest.skip("tabpfn not installed")
    with pytest.warns(UserWarning, match="UNPINNED"):
        clf._build_model()
    assert clf.version_pin == "unpinned"


@tabpfn_required
def test_installed_package_is_the_commercial_v2_line():
    """Guards the licence claim in DESIGN §0.7 against a careless dependency bump."""
    version = installed_tabpfn_version()
    assert version is not None
    assert int(version.split(".")[0]) == 2, f"expected the v2 package line, got {version}"
    assert COMMERCIAL_LICENCE_MARKER in (installed_tabpfn_licence() or "")


@tabpfn_required
def test_fit_context_records_the_pin():
    X, y = _toy()
    clf = CachedTabPFN(config=TabPFNConfig(n_estimators=1)).fit_context(X, y)
    assert clf.fitted
    assert clf.version_pin.startswith("v2")
    assert clf.package_version is not None


@tabpfn_required
def test_context_is_fitted_once_and_reused():
    """Never call .fit() per inference — the context is fixed at commissioning."""
    X, y = _toy()
    clf = CachedTabPFN(config=TabPFNConfig(n_estimators=1)).fit_context(X, y)
    first = clf.predict(X[:10], return_abstain=False)
    model_id = id(clf.model)
    second = clf.predict(X[:10], return_abstain=False)
    assert id(clf.model) == model_id, "model was rebuilt between predictions"
    assert np.array_equal(first.y_pred, second.y_pred)
    # Prediction should be far cheaper than the one-off context fit.
    assert second.latency_predict_s < clf.fit_latency_s


@tabpfn_required
def test_separable_classes_are_classified():
    X, y = _toy(n_per_class=40, n_classes=3)
    clf = CachedTabPFN(config=TabPFNConfig(n_estimators=1)).fit_context(X, y)
    pred = clf.predict(X, return_abstain=False)
    assert (pred.y_pred == y).mean() > 0.9


@tabpfn_required
def test_ood_input_is_abstained_on():
    X, y = _toy(n_per_class=40)
    clf = CachedTabPFN(
        config=TabPFNConfig(n_estimators=1),
        abstention=AbstentionConfig(max_prob_threshold=0.45, enable_mahalanobis=True),
    ).fit_context(X, y)
    far_away = np.full((5, X.shape[1]), 500.0)
    pred = clf.predict(far_away, return_abstain=True)
    assert (pred.y_pred == "ABSTAIN").all()


@tabpfn_required
def test_abstention_can_be_disabled():
    X, y = _toy(n_per_class=40)
    clf = CachedTabPFN(config=TabPFNConfig(n_estimators=1)).fit_context(X, y)
    pred = clf.predict(np.full((5, X.shape[1]), 500.0), return_abstain=False)
    assert not (pred.y_pred == "ABSTAIN").any()


def test_attribution_notice_is_the_licence_required_string():
    """Prior Labs License §10 requires this exact phrase on distribution."""
    assert ATTRIBUTION_NOTICE == "Built with PriorLabs-TabPFN"


def test_context_subsampling_is_class_stratified():
    """A uniform draw would take mostly healthy rows and almost no faults.

    On field data healthy dominates (84% of ESPset), so the reference set has to be
    balanced deliberately or the rare classes vanish from the context entirely.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 5))
    y = np.array(["healthy"] * 900 + ["rare_a"] * 60 + ["rare_b"] * 40)
    clf = CachedTabPFN(config=TabPFNConfig(max_context_rows=150))
    Xs, ys = clf.subsample_context(X, y)
    assert len(ys) == 150
    counts = dict(zip(*np.unique(ys, return_counts=True)))
    # Every class survives, and the rare ones are not crowded out.
    assert set(counts) == {"healthy", "rare_a", "rare_b"}
    assert counts["rare_a"] >= 40 and counts["rare_b"] >= 40


def test_context_subsampling_is_a_noop_below_the_cap():
    X = np.zeros((10, 3))
    y = np.array(["a"] * 5 + ["b"] * 5)
    clf = CachedTabPFN(config=TabPFNConfig(max_context_rows=100))
    Xs, ys = clf.subsample_context(X, y)
    assert len(ys) == 10


def test_context_cap_can_be_disabled():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = np.array(["a", "b"] * 150)
    clf = CachedTabPFN(config=TabPFNConfig(max_context_rows=None))
    Xs, ys = clf.subsample_context(X, y)
    assert len(ys) == 300
