"""MCU-tier anomaly gates: EWMA, CUSUM, Mahalanobis (Cholesky, no inverse)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class EWMAGate:
    """Per-feature EWMA for small sustained shifts (bearing wear, erosion)."""

    lam: float = 0.2
    n_sigma: float = 3.0
    mu: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None

    def fit(self, X_healthy: np.ndarray) -> "EWMAGate":
        X = np.asarray(X_healthy, dtype=float)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-12
        self.z = self.mu.copy()
        return self

    def update(self, x: np.ndarray) -> tuple[bool, np.ndarray]:
        if self.mu is None or self.sigma is None or self.z is None:
            raise RuntimeError("call fit() first")
        x = np.asarray(x, dtype=float)
        self.z = self.lam * x + (1.0 - self.lam) * self.z
        score = np.abs(self.z - self.mu) / self.sigma
        tripped = bool(np.any(score > self.n_sigma))
        return tripped, score


@dataclass
class CUSUM1D:
    """One-sided CUSUM for abrupt mean shift (Page 1954).

    Default direction='down' for dry-run under-current detection.
    """

    k: float = 0.5  # allowance (in sigma units)
    h: float = 5.0  # decision threshold (in sigma units)
    direction: str = "down"  # 'down' | 'up' | 'two'
    mu0: float = 0.0
    sigma: float = 1.0
    s_pos: float = 0.0
    s_neg: float = 0.0

    def fit(self, x_healthy: np.ndarray) -> "CUSUM1D":
        x = np.asarray(x_healthy, dtype=float).ravel()
        self.mu0 = float(x.mean())
        self.sigma = float(x.std()) + 1e-12
        self.reset()
        return self

    def reset(self) -> None:
        self.s_pos = 0.0
        self.s_neg = 0.0

    def update(self, x: float) -> tuple[bool, float]:
        z = (float(x) - self.mu0) / self.sigma
        if self.direction in ("up", "two"):
            self.s_pos = max(0.0, self.s_pos + z - self.k)
        if self.direction in ("down", "two"):
            self.s_neg = max(0.0, self.s_neg - z - self.k)
        score = max(self.s_pos, self.s_neg)
        return score >= self.h, score


@dataclass
class MahalanobisGate:
    """Multivariate gate using precomputed Cholesky factor — never invert Σ on MCU.

    D² = ‖L⁻¹(x − μ)‖² via forward substitution.
    Requires n > 10p healthy samples for a well-conditioned Σ.
    """

    alpha: float = 0.01  # χ² threshold significance
    mu: Optional[np.ndarray] = None
    L: Optional[np.ndarray] = None  # Cholesky of Σ (lower)
    threshold: float = 0.0
    n_features: int = 0

    def fit(self, X_healthy: np.ndarray) -> "MahalanobisGate":
        X = np.asarray(X_healthy, dtype=float)
        n, p = X.shape
        if n < 10 * p:
            raise ValueError(
                f"need n > 10p healthy samples for well-conditioned Σ; got n={n}, p={p}"
            )
        self.n_features = p
        self.mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        # Ridge for numerical stability
        cov = cov + 1e-6 * np.eye(p)
        self.L = np.linalg.cholesky(cov)
        self.threshold = float(stats.chi2.ppf(1.0 - self.alpha, df=p))
        return self

    def distance(self, x: np.ndarray) -> float:
        if self.mu is None or self.L is None:
            raise RuntimeError("call fit() first")
        delta = np.asarray(x, dtype=float) - self.mu
        # Solve L y = delta
        y = np.linalg.solve(self.L, delta)
        return float(y @ y)

    def update(self, x: np.ndarray) -> tuple[bool, float]:
        d2 = self.distance(x)
        return d2 > self.threshold, d2

    def export_baseline(self) -> dict:
        """Ship μ and L to the MCU (no Σ inverse)."""
        if self.mu is None or self.L is None:
            raise RuntimeError("call fit() first")
        return {
            "mu": self.mu.tolist(),
            "L": self.L.tolist(),
            "threshold": self.threshold,
            "n_features": self.n_features,
        }


@dataclass
class CompositeGate:
    """OR of EWMA (drift) + CUSUM on current + Mahalanobis D²."""

    ewma: Optional[EWMAGate] = None
    cusum_current: Optional[CUSUM1D] = None
    mahalanobis: Optional[MahalanobisGate] = None
    current_feature_index: int = -1  # index of current RMS (or ratio) in feature vector

    def update(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float)
        results: dict = {"escalate": False, "reasons": []}

        if self.ewma is not None:
            hit, score = self.ewma.update(x)
            results["ewma_score"] = score
            if hit:
                results["escalate"] = True
                results["reasons"].append("ewma")

        if self.cusum_current is not None and self.current_feature_index >= 0:
            hit, score = self.cusum_current.update(float(x[self.current_feature_index]))
            results["cusum_score"] = score
            if hit:
                results["escalate"] = True
                results["reasons"].append("cusum_current")

        if self.mahalanobis is not None:
            hit, d2 = self.mahalanobis.update(x)
            results["mahalanobis_d2"] = d2
            if hit:
                results["escalate"] = True
                results["reasons"].append("mahalanobis")

        return results
