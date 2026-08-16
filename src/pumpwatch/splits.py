"""Leakage ladder splits (0–4) with explicit in-context reference construction.

Level 0 (random window) is implemented but marked INVALID.
Level 4 (LOMO) is the thesis test for cross-machine generalization.

TabPFN-specific: the context set IS the training set. Leakage into context
is as fatal as leakage into a gradient-trained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

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
) -> SplitResult:
    """Leave-one-group-out. Used for record / component / machine / condition."""
    groups_arr = np.asarray(groups)
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError(f"need >=2 groups for leave-one-out, got {unique}")

    folds = []
    n_tr, n_te = [], []
    for g in unique:
        test_idx = _indices_where(groups_arr == g)
        train_idx = _indices_where(groups_arr != g)
        folds.append(
            SplitFold(
                train_idx=train_idx,
                test_idx=test_idx,
                held_out=str(g),
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
    """Level 4 — leave-one-machine-out. The thesis test."""
    return split_by_group(machine_ids, SplitLevel.LEAVE_ONE_MACHINE_OUT)


def split_record_wise(record_ids: list[str]) -> SplitResult:
    return split_by_group(record_ids, SplitLevel.RECORD_WISE)


def split_component_wise(component_ids: list[str]) -> SplitResult:
    return split_by_group(component_ids, SplitLevel.COMPONENT_WISE)


def split_cross_operating(condition_ids: list[str]) -> SplitResult:
    """Hold out an operating condition (speed/load)."""
    return split_by_group(condition_ids, SplitLevel.CROSS_OPERATING)


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


def normalize_per_machine(
    X: np.ndarray,
    machine_ids: list[str],
    train_idx: np.ndarray,
) -> np.ndarray:
    """Per-machine standardisation fitted on training indices only.

    Global standardisation is silent leakage.
    """
    X = np.asarray(X, dtype=float).copy()
    machines = np.asarray(machine_ids)
    for m_id in set(machines.tolist()):
        tr = train_idx[machines[train_idx] == m_id]
        all_m = np.flatnonzero(machines == m_id)
        if len(tr) == 0:
            continue
        mu = X[tr].mean(axis=0)
        sigma = X[tr].std(axis=0) + 1e-12
        X[all_m] = (X[all_m] - mu) / sigma
    return X
