"""TabPFN v2-pinned classifier with KV-cache-once pattern and abstention.

Pinning, as actually verified against the package rather than assumed:

* PyPI package versions and TabPFN *model* versions are different things, and the
  design document conflates them. There is no 3.x package: PyPI goes 2.x then
  jumps to 6.x/7.x/8.x, where several model versions coexist behind a
  ``ModelVersion`` selector.
* On the 2.x package line there is no ``ModelVersion`` and no
  ``create_default_for_version`` — the package *is* the v2 model, so the package
  constraint ``tabpfn>=2.0,<3`` is itself the pin. Calling
  ``create_default_for_version`` there always raised ImportError, so the previous
  implementation always fell through to an unverified constructor and tagged its
  results "fallback" while warning.
* tabpfn 2.2.1 declares the **Prior Labs License v1.1** = Apache 2.0 plus an
  attribution clause (§10), which is the commercially usable licence DESIGN §0.7
  requires. §10 obliges anyone distributing a product containing the weights to
  display :data:`ATTRIBUTION_NOTICE`.

Both version and licence are checked at construction, and an unverifiable install
raises rather than warning — see :class:`TabPFNVersionError`.

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
    # Refuse to run on a package whose licence cannot be established as
    # commercially usable. Set True only for internal benchmarking.
    allow_unpinned: bool = False
    # "fit_with_cache" initialises the transformer key-value cache during fit(), so
    # repeated predictions against a fixed context skip re-encoding it. This is the
    # KV-cache-at-boot behaviour the design claims; the library default,
    # "fit_preprocessors", caches only preprocessing and leaves the transformer to
    # re-encode the context on every call. Costs memory, which the gateway has and
    # the node does not.
    fit_mode: str = "fit_with_cache"


class TabPFNVersionError(RuntimeError):
    """Raised when the installed TabPFN cannot be pinned to a commercial model."""


# Package versions on the 2.x line ARE the v2 model, and carry the Prior Labs
# License (Apache 2.0 + attribution). Later package lines (6.x+) ship several model
# versions behind a ModelVersion selector, where v2 must be requested explicitly.
V2_PACKAGE_RANGE = (2, 3)
COMMERCIAL_LICENCE_MARKER = "Prior Labs License"

# Prior Labs License §10: distributing a product containing the weights requires
# displaying this. Recorded here so the obligation travels with the code rather
# than living only in a design document.
ATTRIBUTION_NOTICE = "Built with PriorLabs-TabPFN"


def installed_tabpfn_version() -> Optional[str]:
    try:
        import importlib.metadata as md

        return md.version("tabpfn")
    except Exception:
        return None


def installed_tabpfn_licence() -> Optional[str]:
    try:
        import importlib.metadata as md

        lic = md.metadata("tabpfn").get("License")
        return lic.splitlines()[0].strip() if lic else None
    except Exception:
        return None


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
    version_pin: str = "unverified"
    package_version: Optional[str] = None
    licence: Optional[str] = None

    def _build_model(self):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError(
                "tabpfn is required. Install a v2 release: pip install 'tabpfn>=2.0,<3'"
            ) from e

        if self.config.ignore_pretraining_limits:
            os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

        version = installed_tabpfn_version()
        licence = installed_tabpfn_licence()
        self.package_version = version
        self.licence = licence

        # Newer package lines expose several model versions behind a selector; on
        # those, v2 has to be requested explicitly.
        model_version_api = None
        try:
            from tabpfn.constants import ModelVersion  # type: ignore

            model_version_api = ModelVersion
        except Exception:
            pass

        if model_version_api is not None and hasattr(
            TabPFNClassifier, "create_default_for_version"
        ):
            self.version_pin = "v2_via_model_version_api"
            return TabPFNClassifier.create_default_for_version(
                model_version_api.V2,
                n_estimators=self.config.n_estimators,
                device=self.config.device,
                random_state=self.config.random_state,
            )

        # No selector API. The package version itself is then the pin: the 2.x line
        # *is* the v2 model. Verify it rather than assuming — the previous code
        # called create_default_for_version unconditionally, which on a 2.x install
        # always raised ImportError, always fell through to an unverified
        # constructor, and always tagged its results "fallback".
        major = None
        if version:
            try:
                major = int(version.split(".")[0])
            except ValueError:
                major = None

        pinned = major is not None and V2_PACKAGE_RANGE[0] <= major < V2_PACKAGE_RANGE[1]
        commercial = bool(licence and COMMERCIAL_LICENCE_MARKER in licence)

        if not (pinned and commercial) and not self.config.allow_unpinned:
            raise TabPFNVersionError(
                f"Installed tabpfn=={version} (licence: {licence!r}) cannot be "
                f"pinned to the commercially usable v2 model.\n"
                f"Install 'tabpfn>=2.0,<3' (Prior Labs License = Apache 2.0 + "
                f"attribution), or pass TabPFNConfig(allow_unpinned=True) for "
                f"internal benchmarking only — results from an unpinned model must "
                f"not be published as commercially deployable."
            )
        if not (pinned and commercial):
            warnings.warn(
                f"Running UNPINNED tabpfn=={version} (licence: {licence!r}). "
                f"Results are for internal evaluation only.",
                stacklevel=2,
            )
            self.version_pin = "unpinned"
        else:
            self.version_pin = f"v2_package_{version}"

        kwargs = dict(
            n_estimators=self.config.n_estimators,
            device=self.config.device,
            random_state=self.config.random_state,
            ignore_pretraining_limits=self.config.ignore_pretraining_limits,
        )
        if self.config.fit_mode:
            kwargs["fit_mode"] = self.config.fit_mode
        try:
            return TabPFNClassifier(**kwargs)
        except TypeError:
            # Older/newer signatures without fit_mode.
            kwargs.pop("fit_mode", None)
            return TabPFNClassifier(**kwargs)

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

        # Context Mahalanobis for OOD abstention.
        #
        # Requires n > 10p, the same conditioning rule the MCU gate uses. The old
        # condition (n > p + 2) admits a wildly ill-conditioned covariance: with 63
        # features and ~380 context rows the estimated ellipsoid is so badly scaled
        # that every point outside the context lands beyond the chi-squared
        # threshold. Under LOMO that means abstaining on 100% of a new pump's data —
        # silently turning the cross-machine claim into "refuses to answer".
        n_ctx, p_ctx = X_context.shape
        well_conditioned = n_ctx > 10 * p_ctx
        if self.abstention.enable_mahalanobis and not well_conditioned:
            warnings.warn(
                f"OOD abstention disabled: context has n={n_ctx} rows for p={p_ctx} "
                f"features (need n > 10p = {10 * p_ctx}). A covariance estimated from "
                f"too few rows rejects everything.",
                stacklevel=2,
            )
        if self.abstention.enable_mahalanobis and well_conditioned:
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CachedTabPFN":
        """sklearn-compatible alias so TabPFN runs through the same harness as the
        baselines. Named `fit_context` elsewhere because there is no gradient step —
        the rows become the in-context reference set, not training data.
        """
        return self.fit_context(X, y)

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


def benchmark_tabpfn(
    n_context: int = 400,
    n_features: int = 63,
    n_classes: int = 8,
    n_query: int = 64,
    n_repeats: int = 3,
    modes: tuple = ("fit_preprocessors", "fit_with_cache"),
    estimator_counts: tuple = (1, 8),
    seed: int = 0,
) -> list[dict]:
    """Measure fit and predict latency across cache modes and ensemble sizes.

    The design asserts two large latency wins — warming the KV cache once at boot,
    and dropping the ensemble to a single member. Both were quoted from elsewhere
    and neither was enabled in the code: the wrapper used the library default
    ``fit_preprocessors``, which caches preprocessing but leaves the transformer to
    re-encode the whole context on every call. This measures them here instead.
    """
    import time

    rng = np.random.default_rng(seed)
    per_class = max(n_context // n_classes, 2)
    X = np.vstack(
        [rng.normal(3.0 * k, 1.0, (per_class, n_features)) for k in range(n_classes)]
    )
    y = np.array(sum(([f"class{k}"] * per_class for k in range(n_classes)), []))
    X_query = X[rng.choice(len(X), size=min(n_query, len(X)), replace=False)]

    rows = []
    for mode in modes:
        for n_est in estimator_counts:
            clf = CachedTabPFN(
                config=TabPFNConfig(n_estimators=n_est, fit_mode=mode)
            ).fit_context(X, y)
            clf.predict(X_query, return_abstain=False)  # warm-up
            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                clf.predict(X_query, return_abstain=False)
                times.append(time.perf_counter() - t0)
            rows.append({
                "fit_mode": mode,
                "n_estimators": n_est,
                "n_context": int(len(X)),
                "n_features": n_features,
                "n_query": int(len(X_query)),
                "fit_latency_s": clf.fit_latency_s,
                "predict_latency_s": float(np.min(times)),
                "version_pin": clf.version_pin,
            })
    return rows
