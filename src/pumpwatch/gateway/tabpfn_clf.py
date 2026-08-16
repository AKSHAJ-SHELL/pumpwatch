"""TabPFN v2-pinned classifier with KV-cache-once pattern and abstention.

CRITICAL: pip install tabpfn gives v3 (research-only). Pin v2 explicitly via
TabPFNClassifier.create_default_for_version(ModelVersion.V2).

The MCU gate is stage 1. This module runs one multiclass classifier with
abstention/OOD — NOT a redundant healthy-vs-faulty binary TabPFN.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pumpwatch.gateway.baselines import PredResult


@dataclass
class AbstentionConfig:
    """Reject predictions that look OOD relative to the context set."""

    max_prob_threshold: float = 0.45  # below this → abstain
    mahalanobis_alpha: float = 0.01  # χ² gate on feature space vs context
    enable_mahalanobis: bool = True


@dataclass
class TabPFNConfig:
    n_estimators: int = 1  # default 8; 1 is nearly free latency win — validate
    device: str = "cpu"
    ignore_pretraining_limits: bool = True  # allow >1000 on CPU with env flag
    random_state: int = 0


@dataclass
class CachedTabPFN:
    """Fit context once; reuse for all subsequent predict calls.

    Never call .fit() per inference. Context is fixed at commissioning /
    boot of the gateway.
    """

    config: TabPFNConfig = field(default_factory=TabPFNConfig)
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)
    model: object = field(default=None, repr=False)
    classes_: Optional[np.ndarray] = None
    context_mu_: Optional[np.ndarray] = None
    context_L_: Optional[np.ndarray] = None
    context_threshold_: float = 0.0
    fitted: bool = False
    fit_latency_s: float = 0.0
    version_pin: str = "v2"

    def _build_model(self):
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as e:
            raise ImportError(
                "tabpfn is required. Install a v2-compatible release and pin "
                "ModelVersion.V2 — pip's default may be v3 (non-commercial)."
            ) from e

        # Pin v2 for commercial usability (Prior Labs License ≈ Apache 2.0 + attribution)
        if self.config.ignore_pretraining_limits:
            os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

        try:
            model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                n_estimators=self.config.n_estimators,
                device=self.config.device,
                random_state=self.config.random_state,
            )
            self.version_pin = "v2"
        except Exception as exc:
            warnings.warn(
                f"create_default_for_version(V2) failed ({exc}); "
                f"falling back to TabPFNClassifier(n_estimators=...). "
                f"VERIFY the installed package is commercially licensed v2.",
                stacklevel=2,
            )
            model = TabPFNClassifier(
                n_estimators=self.config.n_estimators,
                device=self.config.device,
                random_state=self.config.random_state,
                ignore_pretraining_limits=self.config.ignore_pretraining_limits,
            )
            self.version_pin = "fallback"
        return model

    def fit_context(self, X_context: np.ndarray, y_context: np.ndarray) -> "CachedTabPFN":
        """Warm the context once (KV cache at boot)."""
        X_context = np.asarray(X_context, dtype=np.float64)
        y_context = np.asarray(y_context)
        if X_context.shape[0] > 10_000:
            raise ValueError("TabPFN v2 hard cap is 10_000 training rows")
        if len(np.unique(y_context)) > 10:
            raise ValueError(
                "TabPFN 10-class cap is architectural. Collapse taxonomy or "
                "use a hierarchy outside this classifier — do NOT add a stage-1 "
                "binary TabPFN; the MCU gate is stage 1."
            )

        self.model = self._build_model()
        t0 = time.perf_counter()
        self.model.fit(X_context, y_context)
        self.fit_latency_s = time.perf_counter() - t0
        self.classes_ = np.asarray(self.model.classes_)
        self.fitted = True

        # Context Mahalanobis for OOD abstention
        if self.abstention.enable_mahalanobis and X_context.shape[0] > X_context.shape[1] + 2:
            from scipy import stats

            self.context_mu_ = X_context.mean(axis=0)
            cov = np.cov(X_context, rowvar=False) + 1e-6 * np.eye(X_context.shape[1])
            try:
                self.context_L_ = np.linalg.cholesky(cov)
                self.context_threshold_ = float(
                    stats.chi2.ppf(1.0 - self.abstention.mahalanobis_alpha, df=X_context.shape[1])
                )
            except np.linalg.LinAlgError:
                self.context_L_ = None

        return self

    def _ood_mask(self, X: np.ndarray) -> np.ndarray:
        if self.context_L_ is None or self.context_mu_ is None:
            return np.zeros(len(X), dtype=bool)
        out = np.zeros(len(X), dtype=bool)
        for i, x in enumerate(X):
            y = np.linalg.solve(self.context_L_, x - self.context_mu_)
            d2 = float(y @ y)
            out[i] = d2 > self.context_threshold_
        return out

    def predict(self, X: np.ndarray, return_abstain: bool = True) -> PredResult:
        if not self.fitted or self.model is None:
            raise RuntimeError("call fit_context() once before predict()")
        X = np.asarray(X, dtype=np.float64)
        t0 = time.perf_counter()
        proba = self.model.predict_proba(X)
        y_pred = self.classes_[np.argmax(proba, axis=1)]
        latency = time.perf_counter() - t0

        if return_abstain:
            max_p = proba.max(axis=1)
            abstain = max_p < self.abstention.max_prob_threshold
            abstain |= self._ood_mask(X)
            # Encode abstention as None-like sentinel string
            y_out = y_pred.astype(object)
            y_out[abstain] = "ABSTAIN"
            y_pred = y_out

        return PredResult(
            y_pred=y_pred,
            y_proba=proba,
            classes=self.classes_,
            latency_fit_s=self.fit_latency_s,
            latency_predict_s=latency,
            model_name=f"tabpfn_{self.version_pin}_est{self.config.n_estimators}",
        )


def tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401

        return True
    except ImportError:
        return False
