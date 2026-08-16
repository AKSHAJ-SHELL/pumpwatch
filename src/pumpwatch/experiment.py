"""Shared evaluation harness: leakage ladder + per-fold reporting.

Extracted from scripts/run_experiment.py so that more than one dataset can be put
through identical machinery. The synthetic rig stand-in and ESPset differ in almost
everything — waveforms vs spectra, seeded vs in-service faults, 2 machines vs 11 —
so the one thing that must NOT differ is how they are split and scored. Any
difference in results then belongs to the data rather than the harness.
"""

from __future__ import annotations

import json

import numpy as np

from pumpwatch.evaluate import (
    bootstrap_ci,
    classify_report,
    friedman_nemenyi_allowed,
    reliability_bins,
    risk_coverage_curve,
)
from pumpwatch.gateway.baselines import fit_predict
from pumpwatch.splits import (
    describe_fold,
    normalize_features,
    split_component_wise,
    split_cross_operating,
    split_lomo,
    split_random_window,
    split_record_wise,
)


def _fmt(per_machine: dict, key: str) -> str:
    vals = [m[key] for m in per_machine.values() if m.get(key) is not None]
    return f"{np.mean(vals):.3f}" if vals else "n/a"


def build_ladder(machines, groups, n_samples: int) -> dict[str, object]:
    """Construct every rung of the leakage ladder that this dataset can support.

    A rung whose grouping key is missing or degenerate is omitted rather than
    silently substituted with a weaker split — reporting a "component-wise" number
    that was actually record-wise is exactly the failure the ladder exists to expose.
    """
    ladder: dict[str, object] = {}
    ladder["0_random_window"] = split_random_window(n_samples, seed=0)

    candidates = [
        ("1_record_wise", groups.get("record"), split_record_wise),
        ("2_component_wise", groups.get("component"), split_component_wise),
        ("3_cross_operating", groups.get("operating"), split_cross_operating),
        ("4_lomo", machines, split_lomo),
    ]
    for name, keys, fn in candidates:
        if not keys or len(set(keys)) < 2 or any(k == "" for k in keys):
            print(f"  [skip] {name}: grouping key absent or degenerate")
            continue
        ladder[name] = fn(keys)
    return ladder


def run_split(
    X,
    y,
    machines,
    model_factory,
    model_name: str,
    result,
    norm_strategy: str = "unsupervised_per_machine",
    verbose: bool = False,
) -> dict:
    """Evaluate one rung of the leakage ladder under an explicit normalisation.

    Persists the full per-fold report — PR-AUC (the declared headline metric),
    ROC-AUC, ECE, Brier, coverage, raw confusion counts and latency — not just
    macro-F1. Keeping only macro-F1 is how a repo ends up unable to report the
    metric its own config declares as headline.
    """
    per_machine: dict[str, dict] = {}
    all_true, all_pred = [], []
    fit_latencies, predict_latencies = [], []

    for fold in result.folds:
        Xn = normalize_features(X, machines, fold.train_idx, strategy=norm_strategy)
        model = model_factory()
        pred = fit_predict(
            model,
            Xn[fold.context_idx],
            y[fold.context_idx],
            Xn[fold.test_idx],
            model_name,
        )
        report = classify_report(y[fold.test_idx], pred.y_pred, pred.y_proba, pred.classes)
        entry = {
            **report.as_dict(),
            "fold": describe_fold(fold, machines, y.tolist()),
            "latency_fit_s": pred.latency_fit_s,
            "latency_predict_s": pred.latency_predict_s,
        }
        if pred.y_proba is not None and pred.classes is not None:
            entry["reliability"] = reliability_bins(
                y[fold.test_idx], pred.y_proba, pred.classes
            )
            entry["risk_coverage"] = risk_coverage_curve(
                y[fold.test_idx], pred.y_pred, pred.y_proba.max(axis=1)
            )
        per_machine[fold.held_out] = entry
        all_true.extend(y[fold.test_idx].tolist())
        all_pred.extend(pred.y_pred.tolist())
        fit_latencies.append(pred.latency_fit_s)
        predict_latencies.append(pred.latency_predict_s)
        if verbose:
            print(json.dumps(describe_fold(fold, machines, y.tolist()), indent=2))

    overall = classify_report(np.array(all_true), np.array(all_pred))
    f1s = [m["macro_f1"] for m in per_machine.values()]
    return {
        "model": model_name,
        "norm_strategy": norm_strategy,
        "split_level": int(result.level),
        "split_verdict": result.verdict,
        "per_machine": per_machine,
        "per_machine_macro_f1": {k: v["macro_f1"] for k, v in per_machine.items()},
        "overall_macro_f1": overall.macro_f1,
        "overall_accuracy": overall.accuracy,
        "overall_weighted_f1": overall.weighted_f1,
        "overall_coverage": overall.coverage,
        "overall_confusion": overall.confusion.tolist(),
        "overall_labels": [str(x) for x in overall.labels],
        # Bootstrap at the fold's held-out unit. With 2-3 folds this is
        # near-meaningless and is reported so the reader can see that for themselves.
        "macro_f1_bootstrap_ci": bootstrap_ci(np.array(f1s)),
        "bootstrap_unit": "held_out_group",
        "bootstrap_warning": (
            None if len(f1s) >= 5
            else f"CI over {len(f1s)} folds is not interpretable; report per-fold values"
        ),
        "mean_latency_fit_s": float(np.mean(fit_latencies)),
        "mean_latency_predict_s": float(np.mean(predict_latencies)),
        "n_folds": len(result.folds),
        "friedman_allowed": friedman_nemenyi_allowed(len(result.folds)),
        # Predictions retained so model pairs can be compared with McNemar.
        "_y_true": all_true,
        "_y_pred": all_pred,
    }
