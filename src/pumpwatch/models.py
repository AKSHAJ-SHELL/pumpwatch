"""Canonical model registry — the single place that decides what gets run.

`experiment.py` extracted the evaluation *harness*; this extracts *what is
evaluated*. Before it existed, each of the three experiment scripts built its own
`factories` dict, and the copies drifted:

* the seed fix landed in one script, leaving the other two with zero-arg factories
  that `run_split_repeated` would now reject outright;
* an abstention setting diverged, so a results key called ``tabpfn`` meant an
  abstaining model in one file (coverage 0.81) and a non-abstaining one in another
  (coverage 1.00) — making two published numbers quietly incomparable;
* the torch-before-LightGBM import order, which exists to stop a macOS OpenMP
  segfault, was asserted three times.

Two independent drifts appeared across three copies in about two days of work. The
registry is a response to that base rate: scripts using it cannot disagree, which is
a stronger guarantee than three files that presently happen to match.

Naming rule, and the reason for it: **there is no bare ``tabpfn``**. Abstention
changes what a score means — an abstaining model is graded only on the rows it chose
to answer — so the name has to say which variant produced the number. A results key
must identify a configuration without reference to which script wrote it.
"""

from __future__ import annotations

from typing import Callable, Optional

from pumpwatch.gateway.baselines import (
    MajorityClassifier,
    make_lightgbm,
    make_logistic,
)

# Canonical names. Anything writing a model name into results/ should use these, and
# `test_models.py` asserts the published result files do.
MAJORITY = "majority"
LOGISTIC = "logistic"
LIGHTGBM = "lightgbm"
TABPFN_ABSTAIN = "tabpfn_abstain"
TABPFN_NOABSTAIN = "tabpfn_noabstain"

BASELINE_MODELS = (MAJORITY, LOGISTIC, LIGHTGBM)
TABPFN_MODELS = (TABPFN_ABSTAIN, TABPFN_NOABSTAIN)
ALL_MODELS = BASELINE_MODELS + TABPFN_MODELS

# Models whose output is not affected by the seed. Recorded so a zero spread over
# seeds can be read as "correct" rather than "the seed plumbing is broken again".
DETERMINISTIC_MODELS = (MAJORITY, LOGISTIC)


def lightgbm_available() -> bool:
    """Probe LightGBM once, through make_lightgbm so the OpenMP ordering applies.

    make_lightgbm imports torch first: LightGBM and torch each ship an OpenMP
    runtime and on macOS the process dies (SIGSEGV, exit 139) when the second one
    starts its thread pool. That ordering is load-bearing, and probing through this
    function is what keeps it in one place.
    """
    try:
        make_lightgbm()
        return True
    except ImportError:
        return False


def build_model_zoo(
    include_tabpfn: bool = True,
    tabpfn_context_rows: Optional[int] = 1000,
    n_estimators: int = 1,
    verbose: bool = True,
) -> dict[str, Callable[..., object]]:
    """Return {canonical name: factory}, where every factory accepts ``seed``.

    Seed-awareness is a property of the registry rather than of each caller, so
    ``run_split_repeated``'s guard cannot be tripped by anything built here. The
    guard stays for third-party factories.

    Models unavailable in this environment are omitted rather than stubbed: a
    missing LightGBM should shrink the comparison visibly, not silently substitute
    something else.
    """
    zoo: dict[str, Callable[..., object]] = {
        # Accepts and ignores seed — deterministic, but must share the signature so
        # the harness can treat every factory identically.
        MAJORITY: lambda seed=0: MajorityClassifier(),
        LOGISTIC: lambda seed=0: make_logistic(random_state=seed),
    }

    if lightgbm_available():
        zoo[LIGHTGBM] = lambda seed=0: make_lightgbm(random_state=seed)
    elif verbose:
        print("lightgbm not installed; GBDT baseline omitted")

    if include_tabpfn:
        from pumpwatch.gateway.tabpfn_clf import tabpfn_available

        if tabpfn_available():
            from pumpwatch.gateway.tabpfn_clf import (
                AbstentionConfig,
                CachedTabPFN,
                TabPFNConfig,
            )

            def _config(seed: int) -> "TabPFNConfig":
                return TabPFNConfig(
                    n_estimators=n_estimators,
                    random_state=seed,
                    context_subsample_seed=seed,
                    max_context_rows=tabpfn_context_rows,
                )

            zoo[TABPFN_ABSTAIN] = lambda seed=0: CachedTabPFN(config=_config(seed))
            zoo[TABPFN_NOABSTAIN] = lambda seed=0: CachedTabPFN(
                config=_config(seed),
                abstention=AbstentionConfig(
                    max_prob_threshold=0.0, enable_mahalanobis=False
                ),
            )
        elif verbose:
            print("tabpfn not installed; contributions C2/C4 are UNEVALUATED")

    return zoo


def model_pairs(names) -> list[tuple[str, str]]:
    """Unordered model pairs, for McNemar over every comparison in a run."""
    names = list(names)
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
