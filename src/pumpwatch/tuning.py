"""Nested hyperparameter search that cannot see the held-out machine.

Why this exists: on real ESPset data TabPFN currently beats LightGBM (0.719 vs
0.676 macro-F1 under LOMO), and TabPFN has essentially nothing to tune while the
baselines were run at library defaults. "You didn't tune the baseline" is therefore
a live objection to the headline claim rather than a hypothetical one. If tuned
baselines still lose, the claim gets much stronger; if they don't, that is the
result and it needs to be known before publication, not after.

The hazard is that tuning is itself a way to leak. The outer loop already holds out
a machine; if the inner search sees any row from that machine — directly, or through
a scaler fitted on it — the selected hyperparameters are chosen with knowledge of
the test set and every baseline score is quietly inflated. This module therefore
splits the *training machines only*, by machine, and asserts it.

Grids are deliberately small. A large grid over 11 outer folds multiplies quickly,
and the point is to give the baselines a fair shot, not to win a Kaggle round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Optional

import numpy as np

from pumpwatch.evaluate import classify_report
from pumpwatch.splits import normalize_features


# Small, defensible grids. LightGBM's are the knobs that matter most on small
# tabular data (capacity, step size, leaf-level regularisation); logistic's is the
# only one that does anything here.
LIGHTGBM_GRID = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.05, 0.1],
    "min_child_samples": [5, 20],
}
# Plain `C`, not the Pipeline-style `clf__C`: make_logistic forwards its kwargs
# straight to LogisticRegression rather than to the enclosing Pipeline.
LOGISTIC_GRID = {
    "C": [0.03, 0.1, 1.0, 10.0],
}

DEFAULT_GRIDS = {
    "lightgbm": LIGHTGBM_GRID,
    "logistic": LOGISTIC_GRID,
}


class TuningLeakError(RuntimeError):
    """Raised when an inner search would see the outer fold's held-out machine."""


@dataclass
class TuningResult:
    best_params: dict
    best_score: float
    n_candidates: int
    n_inner_folds: int
    scores: list = field(default_factory=list)
    note: str = ""


def _grid_points(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]


def inner_machine_folds(
    machines: np.ndarray,
    train_idx: np.ndarray,
    max_folds: int = 3,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group-by-machine folds built strictly inside `train_idx`.

    Grouping the inner split by machine too — rather than shuffling rows — means the
    hyperparameters are chosen for cross-machine generalisation, which is what the
    outer loop measures. Tuning on a row-shuffled inner split would select for
    within-machine fit and hand the baselines a worse configuration than they
    deserve.
    """
    train_idx = np.asarray(train_idx)
    m = np.asarray(machines)[train_idx]
    uniq = sorted(set(m.tolist()))
    if len(uniq) < 2:
        return []

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(np.asarray(uniq, dtype=object)))
    k = min(max_folds, len(shuffled))
    bins = [set(shuffled[i::k]) for i in range(k)]

    folds = []
    for held in bins:
        val_mask = np.isin(m, list(held))
        if val_mask.all() or not val_mask.any():
            continue
        folds.append((train_idx[~val_mask], train_idx[val_mask]))
    return folds


def tune_model(
    make_model: Callable[..., object],
    X: np.ndarray,
    y: np.ndarray,
    machines,
    train_idx: np.ndarray,
    grid: dict,
    norm_strategy: str = "train_pooled",
    max_folds: int = 3,
    seed: int = 0,
    held_out_machine: Optional[str] = None,
    scorer: Callable[[np.ndarray, np.ndarray], float] = None,
) -> TuningResult:
    """Grid-search `make_model` inside `train_idx` only.

    `held_out_machine`, when given, is asserted absent from every inner fold. That
    assertion is the whole safety property of this module, so it is checked at run
    time rather than trusted.
    """
    machines_arr = np.asarray(machines)
    train_idx = np.asarray(train_idx)

    if held_out_machine is not None and held_out_machine in set(
        machines_arr[train_idx].tolist()
    ):
        raise TuningLeakError(
            f"held-out machine {held_out_machine!r} appears in the tuning pool; "
            f"the outer split is not actually holding it out"
        )

    candidates = _grid_points(grid)
    folds = inner_machine_folds(machines_arr, train_idx, max_folds=max_folds, seed=seed)
    scorer = scorer or (lambda yt, yp: classify_report(yt, yp).macro_f1)

    if not folds or len(candidates) <= 1:
        return TuningResult(
            best_params=candidates[0] if candidates else {},
            best_score=float("nan"),
            n_candidates=len(candidates),
            n_inner_folds=len(folds),
            note=(
                "no inner tuning performed: "
                + ("only one candidate" if len(candidates) <= 1 else "too few training machines")
            ),
        )

    scored = []
    for params in candidates:
        fold_scores = []
        for inner_train, inner_val in folds:
            if held_out_machine is not None:
                seen = set(machines_arr[inner_train].tolist()) | set(
                    machines_arr[inner_val].tolist()
                )
                if held_out_machine in seen:
                    raise TuningLeakError(
                        f"inner fold contains held-out machine {held_out_machine!r}"
                    )
            # Normalise using the inner training rows only — a scaler fitted across
            # the whole outer training set would leak the inner validation machine.
            Xn = normalize_features(X, machines, inner_train, strategy=norm_strategy)
            model = make_model(**params)
            model.fit(Xn[inner_train], y[inner_train])
            pred = model.predict(Xn[inner_val])
            fold_scores.append(scorer(y[inner_val], np.asarray(pred)))
        scored.append((float(np.mean(fold_scores)), params))

    scored.sort(key=lambda t: -t[0])
    best_score, best_params = scored[0]
    return TuningResult(
        best_params=best_params,
        best_score=best_score,
        n_candidates=len(candidates),
        n_inner_folds=len(folds),
        scores=[{"params": p, "score": s} for s, p in scored],
        note=f"tuned on {len(folds)} machine-grouped inner folds",
    )


def tuned_factory(
    make_model: Callable[..., object],
    grid: dict,
    **tune_kwargs,
) -> Callable[[], object]:
    """Return a zero-arg factory producing a model with tuned hyperparameters.

    Kept separate from :func:`tune_model` so the harness can stay ignorant of
    tuning: `run_split` just receives a factory as it always has.
    """
    result = tune_model(make_model=make_model, grid=grid, **tune_kwargs)

    def factory():
        return make_model(**result.best_params)

    factory.tuning_result = result  # type: ignore[attr-defined]
    return factory
