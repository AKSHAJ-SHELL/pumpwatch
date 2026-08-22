"""Tests for the canonical model registry.

The registry exists to make one class of defect impossible rather than merely
fixed: a results key called `tabpfn` once meant an abstaining model in one file
(coverage 0.81) and a non-abstaining one in another (coverage 1.00), so two
published numbers were quietly incomparable. These tests pin the properties that
prevent that recurring.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from pumpwatch.models import (
    ALL_MODELS,
    BASELINE_MODELS,
    DETERMINISTIC_MODELS,
    LIGHTGBM,
    TABPFN_ABSTAIN,
    TABPFN_NOABSTAIN,
    build_model_zoo,
    lightgbm_available,
    model_pairs,
)


def test_every_factory_accepts_a_seed():
    """run_split_repeated rejects seedless factories; the registry must never
    hand it one."""
    for name, factory in build_model_zoo().items():
        params = inspect.signature(factory).parameters
        assert "seed" in params, f"{name} factory takes no seed"


def test_every_factory_constructs():
    for name, factory in build_model_zoo().items():
        assert factory(seed=0) is not None, name


def test_seed_is_threaded_into_the_model():
    """A factory that accepts a seed and ignores it is the bug in disguise."""
    zoo = build_model_zoo(include_tabpfn=False)
    if LIGHTGBM in zoo:
        assert zoo[LIGHTGBM](seed=7).random_state == 7
        assert zoo[LIGHTGBM](seed=9).random_state == 9


def test_there_is_no_bare_tabpfn_name():
    """The defect this module exists to prevent.

    Abstention changes what a score means, so the name has to say which variant
    produced it.
    """
    assert "tabpfn" not in ALL_MODELS
    assert "tabpfn" not in build_model_zoo()
    assert TABPFN_ABSTAIN in ALL_MODELS and TABPFN_NOABSTAIN in ALL_MODELS


def test_tabpfn_variants_differ_in_abstention():
    zoo = build_model_zoo()
    if TABPFN_ABSTAIN not in zoo:
        pytest.skip("tabpfn not installed")
    abstaining = zoo[TABPFN_ABSTAIN](seed=0)
    always = zoo[TABPFN_NOABSTAIN](seed=0)
    assert abstaining.abstention.enable_mahalanobis is True
    assert always.abstention.enable_mahalanobis is False
    assert always.abstention.max_prob_threshold == 0.0


def test_baselines_are_always_present():
    zoo = build_model_zoo(include_tabpfn=False)
    expected = set(BASELINE_MODELS) if lightgbm_available() else set(BASELINE_MODELS) - {LIGHTGBM}
    assert expected <= set(zoo)


def test_tabpfn_can_be_excluded():
    zoo = build_model_zoo(include_tabpfn=False)
    assert not any(k.startswith("tabpfn") for k in zoo)


def test_deterministic_models_are_declared():
    """So a zero spread over seeds reads as correct rather than as broken plumbing."""
    assert set(DETERMINISTIC_MODELS) <= set(ALL_MODELS)
    assert LIGHTGBM not in DETERMINISTIC_MODELS  # subsampling makes it stochastic


def test_model_pairs_are_unordered_and_complete():
    pairs = model_pairs(["a", "b", "c"])
    assert len(pairs) == 3
    assert ("a", "b") in pairs and ("b", "c") in pairs
    assert ("b", "a") not in pairs


@pytest.mark.parametrize(
    "path", sorted(Path("results").glob("results_*.json")) or [None]
)
def test_published_results_use_registry_names(path):
    """The standing guard: no results file may contain a non-canonical model name.

    This is what makes the naming defect unrepeatable rather than merely fixed —
    a drifted script would have to write an unknown name, and this fails when it does.
    """
    if path is None:
        pytest.skip("no results files present")
    data = json.loads(path.read_text())
    known = set(ALL_MODELS)
    offenders = set()
    for key, val in data.items():
        if not isinstance(val, dict) or "overall_macro_f1" not in val:
            continue
        # Keys are "{model}__{strategy}", "ladder__{rung}__{model}",
        # "{profile}__{model}__{strategy}" or "cross_operating_motor2__{model}".
        parts = key.split("__")
        if any(p in known for p in parts):
            continue
        offenders.add(key)
    assert not offenders, (
        f"{path.name} contains model keys outside the registry: {sorted(offenders)}. "
        "Regenerate it, or add the name to pumpwatch.models."
    )
