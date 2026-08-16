"""Classifier baselines: majority, logistic regression, LightGBM.

These must beat — or expose — TabPFN. Ye et al. (2025) found LR beating
TabPFN in some regimes. Without these the evaluation is not credible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PredResult:
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    classes: np.ndarray
    latency_fit_s: float = 0.0
    latency_predict_s: float = 0.0
    model_name: str = ""


class MajorityClassifier:
    """The floor everyone forgets."""

    def __init__(self) -> None:
        self.majority_: Optional[str] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityClassifier":
        y = np.asarray(y)
        vals, counts = np.unique(y, return_counts=True)
        self.classes_ = vals
        self.majority_ = vals[int(np.argmax(counts))]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.majority_ is None:
            raise RuntimeError("not fitted")
        return np.full(len(X), self.majority_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.majority_ is None:
            raise RuntimeError("not fitted")
        proba = np.zeros((len(X), len(self.classes_)))
        idx = int(np.where(self.classes_ == self.majority_)[0][0])
        proba[:, idx] = 1.0
        return proba


def make_logistic(**kwargs) -> Pipeline:
    params = dict(max_iter=2000, solver="lbfgs")
    params.update(kwargs)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ]
    )


def make_lightgbm(**kwargs):
    try:
        # LightGBM and torch each ship their own OpenMP runtime. On macOS, running
        # both in one process crashes it (SIGSEGV, exit 139) as soon as the second
        # library spins up its thread pool — which is exactly what the TabPFN-vs-GBDT
        # comparison in C4 requires. Importing torch first and holding both to a
        # single thread (n_jobs below, OMP_NUM_THREADS in the runner) avoids the
        # conflicting pools. This is load-bearing, not defensive.
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm is required for GBDT baseline") from e
    params = dict(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        verbose=-1,
        n_jobs=1,
    )
    params.update(kwargs)
    return lgb.LGBMClassifier(**params)


def fit_predict(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model_name: str = "",
) -> PredResult:
    import time

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    t2 = time.perf_counter()

    # CachedTabPFN.predict already returns a fully-formed PredResult (it carries
    # abstention decisions the caller must not lose). Pass it through with the
    # harness-measured latencies attached.
    if isinstance(y_pred, PredResult):
        y_pred.latency_fit_s = t1 - t0
        y_pred.latency_predict_s = t2 - t1
        y_pred.model_name = model_name or y_pred.model_name
        return y_pred

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = model.named_steps["clf"].classes_
    if classes is None:
        classes = np.unique(y_train)
    return PredResult(
        y_pred=np.asarray(y_pred),
        y_proba=y_proba,
        classes=np.asarray(classes),
        latency_fit_s=t1 - t0,
        latency_predict_s=t2 - t1,
        model_name=model_name or type(model).__name__,
    )


def get_baselines(seed: int = 0) -> dict:
    return {
        "majority": MajorityClassifier(),
        "logistic": make_logistic(),
        "lightgbm": make_lightgbm(random_state=seed),
    }
