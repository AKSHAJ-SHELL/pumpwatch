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
    # Trip if ANY of p features exceeds the limit. With p=63 a nominal per-feature
    # 3-sigma limit gives a per-vector false-alarm rate of 1-(0.9973)^63 ~ 16%, so
    # the gate escalates constantly and the two-tier power argument collapses.
    # When set, the per-feature limit is widened (Bonferroni) so the *family* false
    # alarm rate is alpha_family.
    alpha_family: Optional[float] = 0.01
    mu: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None

    def fit(self, X_healthy: np.ndarray) -> "EWMAGate":
        X = np.asarray(X_healthy, dtype=float)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-12
        self.z = self.mu.copy()
        if self.alpha_family is not None:
            p = max(X.shape[1], 1)
            self.n_sigma = float(stats.norm.ppf(1.0 - self.alpha_family / (2.0 * p)))
        return self

    @property
    def sigma_z(self) -> np.ndarray:
        """Steady-state standard deviation of the EWMA statistic itself.

        The smoothed statistic z is far less variable than the raw feature:
        sigma_z = sigma * sqrt(lam / (2 - lam)). Comparing z to the *raw* sigma —
        which this gate used to do — makes a nominal 3-sigma chart roughly a
        9-sigma one at lam=0.2, so the gate is about 3x less sensitive than
        advertised and its false-alarm rate is nowhere near nominal.
        """
        if self.sigma is None:
            raise RuntimeError("call fit() first")
        return self.sigma * np.sqrt(self.lam / (2.0 - self.lam))

    def reset(self) -> None:
        """Return the smoothed statistic to the baseline mean."""
        if self.mu is None:
            raise RuntimeError("call fit() first")
        self.z = self.mu.copy()

    def update(self, x: np.ndarray) -> tuple[bool, np.ndarray]:
        if self.mu is None or self.sigma is None or self.z is None:
            raise RuntimeError("call fit() first")
        x = np.asarray(x, dtype=float)
        self.z = self.lam * x + (1.0 - self.lam) * self.z
        score = np.abs(self.z - self.mu) / self.sigma_z
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
    """OR of EWMA (drift) + CUSUM on current + Mahalanobis D².

    This is stage 1 of the two-tier architecture: it decides which feature vectors
    are worth spending a LoRa transmission on. Its escalation rate is therefore
    the quantity that connects classification accuracy to battery life — see
    :func:`fit_composite_gate` and :func:`evaluate_gate` for the fitted form.
    """

    ewma: Optional[EWMAGate] = None
    cusum_current: Optional[CUSUM1D] = None
    mahalanobis: Optional[MahalanobisGate] = None
    current_feature_index: int = -1  # index of current RMS (or ratio) in feature vector

    def reset(self) -> None:
        if self.ewma is not None:
            self.ewma.reset()
        if self.cusum_current is not None:
            self.cusum_current.reset()

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


# The gate runs on the MCU against a per-pump baseline, so its dimensionality is
# bounded by commissioning length, not by what the feature extractor can compute:
# Mahalanobis needs n > 10p healthy samples, so p=63 would demand ~630 windows of
# healthy operation before the node is armed. These are chosen to span the physical
# failure modes with as few dimensions as possible — overall level, low-frequency
# order content, high-frequency band energy, and load.
DEFAULT_GATE_FEATURES = [
    "iso_vel_rms_mm_s",
    "vib_rms",
    "vib_kurtosis",
    "order_1x",
    "order_2x",
    "band_1_2k",
    "current_rms_ratio",
    "mcsa_sb_1x_sum",
]


def select_gate_features(
    feature_names: list[str],
    wanted: Optional[list[str]] = None,
) -> list[int]:
    """Indices of the gate's feature subset, skipping any the profile lacks."""
    wanted = wanted or DEFAULT_GATE_FEATURES
    return [feature_names.index(w) for w in wanted if w in feature_names]


def fit_composite_gate(
    X_healthy: np.ndarray,
    feature_names: Optional[list[str]] = None,
    current_feature: str = "current_rms_ratio",
    ewma_lam: float = 0.2,
    ewma_n_sigma: float = 3.0,
    cusum_k: float = 0.5,
    cusum_h: float = 5.0,
    mahalanobis_alpha: float = 0.01,
) -> "CompositeGate":
    """Fit all three gate components on healthy commissioning data.

    Resolves the current-feature index by *name*. The dataclass default of -1
    silently disables the CUSUM branch, which meant the advertised
    "EWMA OR CUSUM OR Mahalanobis" composite was really "EWMA OR Mahalanobis"
    anywhere it was constructed by hand.
    """
    X = np.asarray(X_healthy, dtype=float)
    idx = -1
    if feature_names is not None and current_feature in feature_names:
        idx = feature_names.index(current_feature)

    cusum = None
    if idx >= 0:
        cusum = CUSUM1D(k=cusum_k, h=cusum_h, direction="down").fit(X[:, idx])

    maha = None
    if X.shape[0] > 10 * X.shape[1]:
        maha = MahalanobisGate(alpha=mahalanobis_alpha).fit(X)

    return CompositeGate(
        ewma=EWMAGate(lam=ewma_lam, n_sigma=ewma_n_sigma).fit(X),
        cusum_current=cusum,
        mahalanobis=maha,
        current_feature_index=idx,
    )


def evaluate_gate(
    gate: "CompositeGate",
    X: np.ndarray,
    y: np.ndarray,
    healthy_label: str = "healthy",
    field_fault_prevalence: float = 0.01,
) -> dict:
    """Escalation behaviour of stage 1 — the number that links accuracy to battery.

    The gate decides which feature vectors are worth a LoRa transmission. Its
    escalation rate on healthy data sets the radio duty cycle (and therefore
    battery life); its escalation rate on faulty data caps end-to-end recall,
    because the gateway classifier never sees what the gate suppressed. Neither
    number existed before: CompositeGate was never instantiated anywhere.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    gate.reset()

    escalated = np.zeros(len(X), dtype=bool)
    reasons: dict[str, int] = {}
    for i, x in enumerate(X):
        out = gate.update(x)
        escalated[i] = bool(out["escalate"])
        for r in out["reasons"]:
            reasons[r] = reasons.get(r, 0) + 1

    is_healthy = y == healthy_label
    n_healthy = int(is_healthy.sum())
    n_faulty = int((~is_healthy).sum())

    r_healthy = float(escalated[is_healthy].mean()) if n_healthy else float("nan")
    r_faulty = float(escalated[~is_healthy].mean()) if n_faulty else float("nan")
    # The raw escalation rate over a test set is an artefact of how many faulty
    # examples were collected, not a field quantity. Battery life depends on the
    # rate a real node would see, where the pump is healthy almost all the time.
    p = float(field_fault_prevalence)
    field_rate = (1.0 - p) * (r_healthy if n_healthy else 0.0) + p * (
        r_faulty if n_faulty else 0.0
    )

    return {
        "escalation_rate_overall": float(escalated.mean()) if len(X) else 0.0,
        "escalation_rate_field": field_rate,
        "field_fault_prevalence": p,
        # False-escalation: healthy vectors that cost a transmission anyway.
        "escalation_rate_healthy": (
            float(escalated[is_healthy].mean()) if n_healthy else float("nan")
        ),
        # Gate recall: the ceiling on end-to-end fault detection.
        "escalation_rate_faulty": (
            float(escalated[~is_healthy].mean()) if n_faulty else float("nan")
        ),
        "n_healthy": n_healthy,
        "n_faulty": n_faulty,
        "reasons": reasons,
    }
