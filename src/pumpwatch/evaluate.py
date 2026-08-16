"""Evaluation harness: PR-AUC, recall@fixed-FAR, calibration, McNemar, bootstrap.

Right-sized stats: no Friedman–Nemenyi at n=3 machines. Machine-level
bootstrap only when enough machines exist; otherwise recording-level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion: np.ndarray
    labels: list
    pr_auc_macro: Optional[float] = None
    ece: Optional[float] = None
    brier: Optional[float] = None


def safe_macro_pr_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Optional[float]:
    y_true = np.asarray(y_true)
    # Filter abstentions
    mask = y_true != "ABSTAIN"
    # Also drop rows where pred was abstain if y_true aligned differently — caller handles
    scores = []
    for i, c in enumerate(classes):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(average_precision_score(yt, y_proba[:, i]))
    return float(np.mean(scores)) if scores else None


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Multiclass ECE on max predicted probability."""
    y_true = np.asarray(y_true)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    # Map labels to indices; skip unknowns
    conf = y_proba.max(axis=1)
    pred_idx = y_proba.argmax(axis=1)
    correct = np.array(
        [
            class_to_idx.get(yt, -1) == pi
            for yt, pi in zip(y_true, pred_idx)
        ],
        dtype=float,
    )
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if not np.any(m):
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> dict:
    conf = y_proba.max(axis=1)
    pred_idx = y_proba.argmax(axis=1)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    correct = np.array(
        [class_to_idx.get(yt, -1) == pi for yt, pi in zip(y_true, pred_idx)],
        dtype=float,
    )
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf, bin_acc, bin_count = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        bin_count.append(int(m.sum()))
        bin_conf.append(float(conf[m].mean()) if m.any() else float("nan"))
        bin_acc.append(float(correct[m].mean()) if m.any() else float("nan"))
    return {"bin_conf": bin_conf, "bin_acc": bin_acc, "bin_count": bin_count}


def recall_at_fixed_far(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
    max_far: float,
) -> dict:
    """Recall at a false-alarm rate ceiling (e.g. ≤1 false alarm / pump / month).

    y_true_binary: 1 = fault, 0 = healthy.
    scores: higher = more faulty.
    """
    y = np.asarray(y_true_binary).astype(int)
    s = np.asarray(scores, dtype=float)
    # Sweep thresholds from high to low
    thresholds = np.unique(s)[::-1]
    best = {"recall": 0.0, "far": 0.0, "threshold": float("inf")}
    n_neg = max(int((y == 0).sum()), 1)
    n_pos = max(int((y == 1).sum()), 1)
    for thr in thresholds:
        pred = (s >= thr).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        far = fp / n_neg
        rec = tp / n_pos
        if far <= max_far and rec >= best["recall"]:
            best = {"recall": rec, "far": far, "threshold": float(thr)}
    return best


def classify_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
) -> ClassificationReport:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Drop abstentions from scoring
    mask = (y_pred != "ABSTAIN") & (y_true != "ABSTAIN")
    yt, yp = y_true[mask], y_pred[mask]
    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    cm = confusion_matrix(yt, yp, labels=labels)
    report = ClassificationReport(
        accuracy=float(accuracy_score(yt, yp)) if len(yt) else 0.0,
        macro_f1=float(f1_score(yt, yp, average="macro", zero_division=0)) if len(yt) else 0.0,
        weighted_f1=float(f1_score(yt, yp, average="weighted", zero_division=0)) if len(yt) else 0.0,
        confusion=cm,
        labels=labels,
    )
    if y_proba is not None and classes is not None:
        yp_full = y_proba[mask] if mask.shape == y_true.shape else y_proba
        # Align mask length
        if len(yp_full) == len(y_true):
            yp_full = y_proba[mask]
            yt_full = y_true[mask]
        else:
            yt_full = yt
            yp_full = y_proba
        report.pr_auc_macro = safe_macro_pr_auc(yt_full, yp_full, classes)
        report.ece = expected_calibration_error(yt_full, yp_full, classes)
        # Brier on one-vs-rest max class — approximate multiclass Brier
        try:
            # Use probability of true class
            class_to_idx = {c: i for i, c in enumerate(classes)}
            p_true = np.array(
                [yp_full[i, class_to_idx[t]] if t in class_to_idx else 0.0 for i, t in enumerate(yt_full)]
            )
            report.brier = float(np.mean((1.0 - p_true) ** 2))
        except Exception:
            report.brier = None
    return report


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar exact binomial test (Dietterich 1998). Prefer over resampled t-test."""
    y_true = np.asarray(y_true)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    n01 = int((~a_correct & b_correct).sum())  # A wrong, B right
    n10 = int((a_correct & ~b_correct).sum())  # A right, B wrong
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0, "method": "exact"}
    # Exact binomial two-sided
    p = float(stats.binomtest(n01, n, 0.5).pvalue)
    return {"n01": n01, "n10": n10, "p_value": p, "method": "exact"}


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI. Resample at the provided unit (recording/machine)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(sample.mean())
    boots = np.sort(boots)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"mean": float(values.mean()), "lo": lo, "hi": hi}


def friedman_nemenyi_allowed(n_datasets: int) -> bool:
    """CD diagrams need >=5 datasets. LOMO with 2–3 machines does not qualify."""
    return n_datasets >= 5
