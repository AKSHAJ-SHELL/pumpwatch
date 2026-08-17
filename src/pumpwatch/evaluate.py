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
    confusion_matrix,
    f1_score,
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
    roc_auc_macro: Optional[float] = None
    ece: Optional[float] = None
    brier: Optional[float] = None
    # Selective prediction. Every metric above is computed on the covered rows only,
    # so a model that abstains is scored on an easier subset than one that does not.
    # Comparing an abstaining model to a non-abstaining baseline without reporting
    # coverage silently flatters the abstainer.
    coverage: float = 1.0
    n_total: int = 0
    n_scored: int = 0
    n_abstained: int = 0

    def as_dict(self) -> dict:
        """JSON-safe summary — confusion matrix as raw counts, per DESIGN §5."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "pr_auc_macro": self.pr_auc_macro,
            "roc_auc_macro": self.roc_auc_macro,
            "ece": self.ece,
            "brier": self.brier,
            "coverage": self.coverage,
            "n_total": self.n_total,
            "n_scored": self.n_scored,
            "n_abstained": self.n_abstained,
            "labels": [str(x) for x in self.labels],
            "confusion": self.confusion.tolist(),
        }


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    n_points: int = 20,
) -> dict:
    """Error rate as a function of coverage, sweeping a confidence threshold.

    Lets an abstaining model be compared to a non-abstaining one at matched
    coverage rather than at its own self-selected operating point.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    conf = np.asarray(confidence, dtype=float)
    order = np.argsort(-conf)  # most confident first
    correct = (y_pred[order] == y_true[order]).astype(float)
    n = len(correct)
    if n == 0:
        return {"coverage": [], "risk": [], "threshold": []}
    cutoffs = np.unique(np.linspace(1, n, min(n_points, n)).astype(int))
    cov, risk, thr = [], [], []
    for k in cutoffs:
        cov.append(float(k) / n)
        risk.append(float(1.0 - correct[:k].mean()))
        thr.append(float(conf[order][k - 1]))
    return {"coverage": cov, "risk": risk, "threshold": thr}


def safe_macro_pr_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Optional[float]:
    y_true = np.asarray(y_true)
    scores = []
    for i, c in enumerate(classes):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(average_precision_score(yt, y_proba[:, i]))
    return float(np.mean(scores)) if scores else None


def safe_macro_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Optional[float]:
    """Macro one-vs-rest ROC-AUC.

    Secondary to PR-AUC by design (DESIGN §5): on imbalanced fault data ROC-AUC's
    false-positive denominator is the large healthy class, so it stays deceptively
    high. Reported anyway because it is prevalence-independent, which matters when
    test prevalence is an artefact of how data was collected rather than a real rate.
    """
    y_true = np.asarray(y_true)
    scores = []
    for i, c in enumerate(classes):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(roc_auc_score(yt, y_proba[:, i]))
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
    n_total = len(y_true)

    # Abstentions are excluded from scoring, but the exclusion is recorded — see
    # ClassificationReport.coverage.
    keep = (y_pred.astype(str) != "ABSTAIN") & (y_true.astype(str) != "ABSTAIN")
    yt, yp = y_true[keep], y_pred[keep]
    n_scored = len(yt)

    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    cm = confusion_matrix(yt, yp, labels=labels) if n_scored else np.zeros((0, 0), dtype=int)
    report = ClassificationReport(
        accuracy=float(accuracy_score(yt, yp)) if n_scored else 0.0,
        macro_f1=float(f1_score(yt, yp, average="macro", zero_division=0)) if n_scored else 0.0,
        weighted_f1=float(f1_score(yt, yp, average="weighted", zero_division=0)) if n_scored else 0.0,
        confusion=cm,
        labels=labels,
        coverage=float(n_scored) / n_total if n_total else 0.0,
        n_total=n_total,
        n_scored=n_scored,
        n_abstained=n_total - n_scored,
    )

    if y_proba is not None and classes is not None and n_scored:
        y_proba = np.asarray(y_proba, dtype=float)
        # y_proba is emitted per input row, so it aligns with y_true before masking.
        proba = y_proba[keep] if len(y_proba) == n_total else y_proba
        if len(proba) != n_scored:
            return report  # cannot align — leave probability metrics unset
        report.pr_auc_macro = safe_macro_pr_auc(yt, proba, classes)
        report.roc_auc_macro = safe_macro_roc_auc(yt, proba, classes)
        report.ece = expected_calibration_error(yt, proba, classes)
        report.brier = multiclass_brier(yt, proba, classes)
    return report


def multiclass_brier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> Optional[float]:
    """Multiclass Brier score: mean over samples of sum_k (p_k - onehot_k)^2.

    The one-sided (1 - p_true)^2 form ignores how the remaining mass is distributed
    and so cannot distinguish a confident wrong answer from a diffuse one.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx = np.array([class_to_idx.get(t, -1) for t in y_true])
    if np.any(idx < 0):
        return None
    onehot = np.zeros_like(y_proba)
    onehot[np.arange(len(idx)), idx] = 1.0
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))


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


# Windows a node actually produces per month, from the event-triggered energy model
# (feature_compute_per_runtime_hour=12 at 3 h/day runtime). This is the bridge
# between the farmer-facing statement and the metric: "at most one false alarm per
# pump per month" is only a number once you know how many decisions get made.
DEFAULT_WINDOWS_PER_RUNTIME_HOUR = 12.0
DEFAULT_RUNTIME_HOURS_PER_DAY = 3.0


def windows_per_month(
    windows_per_runtime_hour: float = DEFAULT_WINDOWS_PER_RUNTIME_HOUR,
    runtime_hours_per_day: float = DEFAULT_RUNTIME_HOURS_PER_DAY,
    days: float = 30.0,
) -> float:
    return windows_per_runtime_hour * runtime_hours_per_day * days


def far_for_alarms_per_month(
    alarms_per_month: float = 1.0,
    **kwargs,
) -> float:
    """Convert an alarms-per-pump-per-month budget into a per-window FAR.

    At the default duty (12 windows/runtime-hour, 3 h/day, 30 days) a node makes
    ~1080 decisions a month, so "one false alarm a month" is a per-window
    false-alarm rate of ~0.001 — a far harsher target than the 0.01 that a
    generic ROC operating point would suggest.
    """
    n = windows_per_month(**kwargs)
    if n <= 0:
        raise ValueError("windows per month must be positive")
    return float(alarms_per_month) / n


def fault_score_from_proba(
    y_proba: np.ndarray,
    classes: np.ndarray,
    healthy_label: str = "healthy",
) -> np.ndarray:
    """P(not healthy) — the natural detector score for a multiclass classifier.

    Turns the multiclass output into the binary decision the node actually makes
    (escalate or not) without collapsing the classifier itself to binary.
    """
    classes = np.asarray(classes)
    idx = np.flatnonzero(classes == healthy_label)
    if len(idx) == 0:
        return np.ones(len(y_proba))
    return 1.0 - np.asarray(y_proba, dtype=float)[:, idx[0]]


def recall_at_alarm_budget(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    healthy_label: str = "healthy",
    alarms_per_month: float = 1.0,
    **duty_kwargs,
) -> dict:
    """Fault recall at a stated false-alarm budget, in the farmer's units.

    DESIGN §5 makes this the metric that matters operationally: the costs are
    asymmetric (a missed dry-run destroys a seal, a false alarm wastes a trip), so
    an F1 score answers the wrong question. Reported with the FAR it was measured
    at and the duty assumption behind it, because the number is meaningless without
    both.
    """
    y_true = np.asarray(y_true)
    scores = fault_score_from_proba(y_proba, classes, healthy_label)
    is_fault = (y_true != healthy_label).astype(int)
    max_far = far_for_alarms_per_month(alarms_per_month, **duty_kwargs)
    best = recall_at_fixed_far(is_fault, scores, max_far)
    return {
        **best,
        "max_far": max_far,
        "alarms_per_month_budget": alarms_per_month,
        "windows_per_month": windows_per_month(**duty_kwargs),
        "n_fault": int(is_fault.sum()),
        "n_healthy": int((1 - is_fault).sum()),
    }


def detection_by_severity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    severity: np.ndarray,
    healthy_label: str = "healthy",
) -> dict:
    """Detection rate as a function of fault severity.

    The honest substitute for a lead-time curve when no run-to-failure data
    exists. Twente grades its faults (bearing bpfo 1/2/3, cavitation suction
    1/3/4, align angular 1-5), and while a severity ordering is not a time axis,
    "how far must the fault progress before we see it" is the question lead time
    was being used to answer. Reporting this instead of a fabricated RUL curve is
    the defensible move (DESIGN §8.3 option b).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sev = np.asarray(severity, dtype=object)

    is_fault = y_true != healthy_label
    detected = (y_pred != healthy_label) & (y_pred.astype(str) != "ABSTAIN")

    out = {}
    for s in sorted({v for v in sev[is_fault] if v is not None}):
        m = is_fault & (sev == s)
        n = int(m.sum())
        if n == 0:
            continue
        out[str(s)] = {
            "n": n,
            "detected_rate": float(detected[m].mean()),
            "correct_class_rate": float((y_pred[m] == y_true[m]).mean()),
        }
    return {
        "by_severity": out,
        "note": (
            "Severity index is the dataset's own grading, not elapsed time. A "
            "rising detection rate means the indicator responds before the fault "
            "is fully developed; it does NOT license a remaining-useful-life claim."
        ),
    }
