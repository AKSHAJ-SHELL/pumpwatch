"""Leakage ladder splits (0–4) with explicit in-context reference construction.

Level 0 (random window) is implemented but marked INVALID.
Level 4 (LOMO) is the thesis test for cross-machine generalization.

TabPFN-specific: the context set IS the training set. Leakage into context
is as fatal as leakage into a gradient-trained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class SplitLevel(IntEnum):
    RANDOM_WINDOW = 0  # INVALID — memorisation test
    RECORD_WISE = 1
    COMPONENT_WISE = 2
    CROSS_OPERATING = 3
    LEAVE_ONE_MACHINE_OUT = 4


LEVEL_VERDICT = {
    SplitLevel.RANDOM_WINDOW: "INVALID",
    SplitLevel.RECORD_WISE: "weak",
    SplitLevel.COMPONENT_WISE: "good",
    SplitLevel.CROSS_OPERATING: "essential",
    SplitLevel.LEAVE_ONE_MACHINE_OUT: "thesis_test",
}


@dataclass
class SplitFold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    held_out: str
    level: SplitLevel
    # Explicit context construction for TabPFN
    context_idx: np.ndarray  # same as train_idx, named for clarity


@dataclass
class SplitResult:
    level: SplitLevel
    folds: list[SplitFold]
    verdict: str
    n_machines_train: list[int]
    n_machines_test: list[int]


def _indices_where(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def split_random_window(
    n: int,
    test_frac: float = 0.3,
    seed: int = 0,
) -> SplitResult:
    """Level 0 — INVALID. Exists only to show the leakage ladder collapse."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    fold = SplitFold(
        train_idx=train_idx,
        test_idx=test_idx,
        held_out="random_windows",
        level=SplitLevel.RANDOM_WINDOW,
        context_idx=train_idx,
    )
    return SplitResult(
        level=SplitLevel.RANDOM_WINDOW,
        folds=[fold],
        verdict=LEVEL_VERDICT[SplitLevel.RANDOM_WINDOW],
        n_machines_train=[0],
        n_machines_test=[0],
    )


def split_by_group(
    groups: list[str],
    level: SplitLevel,
    seed: int = 0,
    max_folds: int = 10,
) -> SplitResult:
    """Leave-one-group-out. Used for record / component / machine / condition.

    When there are more groups than ``max_folds``, groups are partitioned into
    ``max_folds`` disjoint bins and each bin is held out in turn — grouped k-fold
    rather than leave-one-out. This keeps the leakage guarantee exactly (no group
    ever spans train and test) while keeping a record-wise split over hundreds of
    recordings computationally sane. Set ``max_folds=0`` for true leave-one-out.
    """
    groups_arr = np.asarray(groups)
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError(f"need >=2 groups for leave-one-out, got {unique}")

    if max_folds and len(unique) > max_folds:
        rng = np.random.default_rng(seed)
        shuffled = list(rng.permutation(np.asarray(unique, dtype=object)))
        bins = [shuffled[i::max_folds] for i in range(max_folds)]
        held_out_sets = [(f"bin{i}({len(b)} groups)", set(b)) for i, b in enumerate(bins) if b]
    else:
        held_out_sets = [(str(g), {g}) for g in unique]

    folds = []
    n_tr, n_te = [], []
    for name, held in held_out_sets:
        mask = np.isin(groups_arr, list(held))
        test_idx = _indices_where(mask)
        train_idx = _indices_where(~mask)
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        folds.append(
            SplitFold(
                train_idx=train_idx,
                test_idx=test_idx,
                held_out=name,
                level=level,
                context_idx=train_idx.copy(),
            )
        )
        n_tr.append(len(set(groups_arr[train_idx].tolist())))
        n_te.append(len(set(groups_arr[test_idx].tolist())))

    return SplitResult(
        level=level,
        folds=folds,
        verdict=LEVEL_VERDICT[level],
        n_machines_train=n_tr,
        n_machines_test=n_te,
    )


def split_lomo(machine_ids: list[str]) -> SplitResult:
    """Level 4 — leave-one-machine-out. The thesis test.

    Never binned: the machine IS the experimental unit here, so every fold holds
    out exactly one. Binning machines together to cap fold count (which the generic
    grouped split does for record-wise splits over hundreds of sessions) would
    destroy the per-machine breakdown that the claim rests on, and would silently
    turn an 11-point result into a 10-point one.
    """
    return split_by_group(machine_ids, SplitLevel.LEAVE_ONE_MACHINE_OUT, max_folds=0)


def split_record_wise(record_ids: list[str]) -> SplitResult:
    return split_by_group(record_ids, SplitLevel.RECORD_WISE)


def split_component_wise(component_ids: list[str]) -> SplitResult:
    return split_by_group(component_ids, SplitLevel.COMPONENT_WISE)


def split_cross_operating(condition_ids: list[str]) -> SplitResult:
    """Hold out an operating condition (speed/load)."""
    return split_by_group(condition_ids, SplitLevel.CROSS_OPERATING)


def split_label_coverage(result: SplitResult, labels: list[str]) -> dict:
    """Check every fold trains on the classes it will be tested on.

    A split is only interpretable if the training side of each fold contains the
    classes that appear on its test side. When it does not, the model is asked for
    a label it has never seen, scores zero on it by construction, and the resulting
    macro-F1 measures the split's degeneracy rather than the model.

    This is the general form of the Twente leave-one-machine-out problem: there the
    two motors carry disjoint fault sets, so LOMO is meaningless. The same thing
    happens to a component-wise split when a fault family has only one physical
    component in the loaded subset, and to a cross-operating split when a condition
    was only ever run at one speed. Cheap to check, and silent if not checked.
    """
    y = np.asarray(labels)
    per_fold = []
    total_missing = 0
    for fold in result.folds:
        train_classes = set(y[fold.train_idx].tolist())
        test_classes = set(y[fold.test_idx].tolist())
        missing = sorted(test_classes - train_classes)
        total_missing += len(missing)
        per_fold.append({
            "held_out": fold.held_out,
            "n_test_classes": len(test_classes),
            "missing_from_train": missing,
        })
    n_test_classes = sum(f["n_test_classes"] for f in per_fold) or 1
    frac = total_missing / n_test_classes
    return {
        "level": int(result.level),
        "verdict": result.verdict,
        "n_folds": len(result.folds),
        "folds": per_fold,
        "total_unseen_test_classes": total_missing,
        "fraction_test_classes_unseen": frac,
        # Any unseen class distorts macro-F1; a large share makes the rung useless.
        "interpretable": frac < 0.10,
        "note": (
            "Folds testing on classes absent from their own training set score zero "
            "on those classes by construction. Treat the rung as a statement about "
            "the data's structure, not the model's ability."
        ),
    }


def describe_fold(fold: SplitFold, machine_ids: list[str], classes: list[str]) -> dict:
    """Counts a careful reader wants: unique pumps/classes in context vs test."""
    m = np.asarray(machine_ids)
    c = np.asarray(classes)
    return {
        "held_out": fold.held_out,
        "level": int(fold.level),
        "verdict": LEVEL_VERDICT[fold.level],
        "n_context": int(len(fold.context_idx)),
        "n_test": int(len(fold.test_idx)),
        "machines_context": sorted(set(m[fold.context_idx].tolist())),
        "machines_test": sorted(set(m[fold.test_idx].tolist())),
        "classes_context": sorted(set(c[fold.context_idx].tolist())),
        "classes_test": sorted(set(c[fold.test_idx].tolist())),
    }


NORMALIZATION_STRATEGIES = ("unsupervised_per_machine", "train_pooled")


def _standardise(block: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Z-score `block` using statistics from `fit_rows`.

    Zero-variance features are left centred but unscaled — dividing by ~0 turns a
    constant feature into an arbitrarily large one and swamps every real feature.
    """
    mu = fit_rows.mean(axis=0)
    sigma = fit_rows.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return (block - mu) / sigma


def normalize_features(
    X: np.ndarray,
    machine_ids: list[str],
    train_idx: np.ndarray,
    strategy: str = "unsupervised_per_machine",
) -> np.ndarray:
    """Standardise features under an explicit, stated normalisation strategy.

    Under LOMO the held-out machine has **no labelled rows**, so "fit the scaler on
    training data" is not defined for it. That is a real deployment question — a pump
    commissioned today has no labelled history — not an implementation detail, so the
    caller must choose which assumption to make and the paper must state it:

    ``train_pooled``
        One scaler fitted on the pooled training machines, applied to everything.
        Fully inductive: nothing about the test machine is used. The conservative
        baseline, and the honest number if you claim zero-touch commissioning.

    ``unsupervised_per_machine``
        Each machine is standardised by its **own** statistics. For the held-out
        machine those come from its own rows — which uses no labels, only the
        unlabelled feature distribution the node would collect during commissioning.
        Transductive, and must be declared as such.

    The gap between the two is itself a result about cross-machine transfer: it
    measures how much of the LOMO score depends on seeing the target pump's
    operating distribution at all.

    Global standardisation over train+test pooled together is silent leakage and is
    deliberately not offered.
    """
    if strategy not in NORMALIZATION_STRATEGIES:
        raise ValueError(
            f"unknown normalisation strategy {strategy!r}; "
            f"expected one of {NORMALIZATION_STRATEGIES}"
        )
    X = np.asarray(X, dtype=float).copy()
    machines = np.asarray(machine_ids)
    train_idx = np.asarray(train_idx)

    if strategy == "train_pooled":
        if len(train_idx) == 0:
            raise ValueError("train_pooled needs at least one training row")
        return _standardise(X, X[train_idx])

    # unsupervised_per_machine
    for m_id in set(machines.tolist()):
        all_m = np.flatnonzero(machines == m_id)
        tr = train_idx[machines[train_idx] == m_id]
        # Held-out machine: fit on its own unlabelled rows. Never skip — leaving a
        # machine in raw units while the rest are z-scored guarantees a scale
        # mismatch and collapses every model to chance.
        fit_rows = X[tr] if len(tr) > 0 else X[all_m]
        X[all_m] = _standardise(X[all_m], fit_rows)
    return X
