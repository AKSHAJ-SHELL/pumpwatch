#!/usr/bin/env python3
"""Run core experiments: synth → audit → baselines → (optional TabPFN) → LOMO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure src on path when run as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pumpwatch.audit import assert_not_confounded, audit_confound
from pumpwatch.datasets.twente import write_demo_twente_cache, load_twente
from pumpwatch.evaluate import (
    bootstrap_ci,
    classify_report,
    friedman_nemenyi_allowed,
    mcnemar_exact,
    reliability_bins,
    risk_coverage_curve,
)
from pumpwatch.features import FeatureMeta, extract_features, feature_matrix
from pumpwatch.gateway.baselines import MajorityClassifier, fit_predict, get_baselines
from pumpwatch.gateway.tabpfn_clf import CachedTabPFN, TabPFNConfig, tabpfn_available
from pumpwatch.node.energy import event_triggered_energy, fixed_schedule_energy
from pumpwatch.node.trip import evaluate_trip_path
from pumpwatch.physics import BearingGeometry
from pumpwatch.splits import (
    NORMALIZATION_STRATEGIES,
    describe_fold,
    normalize_features,
    split_lomo,
)
from pumpwatch.synth import Condition, generate_dataset


def build_feature_table(records, profile: str):
    vecs, y, machines, sources = [], [], [], []
    for r in records:
        # SynthRecord or Twente-like
        if hasattr(r, "condition") and hasattr(r.condition, "value"):
            label = r.condition.value
            pump_id = r.meta.pump_id
            vib = r.vibration
            fs = r.fs
            current = r.current_rms
            current_wave = r.current_waveform
            fs_current = r.fs
            rpm = r.meta.rpm
            rated = r.meta.rated_current_a
            n_vanes = r.meta.n_vanes
            bearing = r.meta.bearing
            src = "synth"
        else:
            label = r.condition
            pump_id = r.pump_id
            vib = r.vibration
            fs = r.fs_vib
            current = r.current
            current_wave = r.current_waveform
            fs_current = r.fs_current
            rpm = r.rpm
            # Geometry comes from the manifest. Hardcoding it (this used to be
            # rated=10.0, n_vanes=6, BearingGeometry(8, 7.0, 35.0) for every record)
            # puts every VPF and bearing-envelope feature at an invented frequency.
            rated = r.rated_current_a
            n_vanes = r.n_vanes
            bearing = r.bearing_geometry()
            src = r.source

        meta = FeatureMeta(
            rpm=rpm,
            n_vanes=n_vanes,
            # Geometry is nameplate data, not a sensor — it is just as knowable for a
            # borewell pump as a surface one, and MCSA bearing sidebands ride on the
            # current channel. The profile decides which *signals* exist, not which
            # facts about the pump are known.
            bearing=bearing,
            rated_current_a=rated,
            voltage_available=False,
            profile=profile,
        )
        fv = extract_features(
            vib,
            fs,
            current_rms=current,
            current_waveform=current_wave,
            fs_current=fs_current,
            meta=meta,
        )
        vecs.append(fv)
        y.append(label)
        machines.append(pump_id)
        sources.append(src)
    X, names = feature_matrix(vecs)
    return X, np.array(y), machines, sources, names


def _fmt(per_machine: dict, key: str) -> str:
    vals = [m[key] for m in per_machine.values() if m.get(key) is not None]
    return f"{np.mean(vals):.3f}" if vals else "n/a"


def run_lomo(
    X,
    y,
    machines,
    model_factory,
    model_name: str,
    norm_strategy: str = "unsupervised_per_machine",
    verbose: bool = False,
) -> dict:
    """Leave-one-machine-out under an explicit normalisation strategy.

    Persists the full per-fold report — PR-AUC (the declared headline metric),
    ROC-AUC, ECE, Brier, coverage, raw confusion counts and latency — not just
    macro-F1. Keeping only macro-F1 is how a repo ends up unable to report the
    metric its own config declares as headline.
    """
    result = split_lomo(machines)
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
        "per_machine": per_machine,
        "per_machine_macro_f1": {k: v["macro_f1"] for k, v in per_machine.items()},
        "overall_macro_f1": overall.macro_f1,
        "overall_accuracy": overall.accuracy,
        "overall_weighted_f1": overall.weighted_f1,
        "overall_coverage": overall.coverage,
        "overall_confusion": overall.confusion.tolist(),
        "overall_labels": [str(x) for x in overall.labels],
        # Bootstrap at the machine level — the thesis unit. With 2-3 machines this is
        # near-meaningless and is reported so the reader can see that for themselves.
        "macro_f1_bootstrap_ci": bootstrap_ci(np.array(f1s)),
        "bootstrap_unit": "machine",
        "bootstrap_warning": (
            None if len(f1s) >= 5
            else f"CI over {len(f1s)} machines is not interpretable; report per-machine values"
        ),
        "mean_latency_fit_s": float(np.mean(fit_latencies)),
        "mean_latency_predict_s": float(np.mean(predict_latencies)),
        "n_folds": len(result.folds),
        "friedman_allowed": friedman_nemenyi_allowed(len(result.folds)),
        # Predictions retained so model pairs can be compared with McNemar.
        "_y_true": all_true,
        "_y_pred": all_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "ct_only"], default="full")
    parser.add_argument("--demo-twente", type=Path, default=ROOT / "data" / "twente_demo")
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=== Energy sanity ===")
    e_evt = event_triggered_energy(3.0)
    e_fix = fixed_schedule_energy(900.0)
    print(f"event-triggered 3h/day: {e_evt.mAh_per_day:.2f} mAh/day, {e_evt.battery_years:.2f} yr")
    print(f"fixed 15min (FALSIFIED): {e_fix.mAh_per_day:.2f} mAh/day — {e_fix.notes}")

    print("\n=== Dry-run trip path ===")
    trip = evaluate_trip_path(n_trials=30, seed=0)
    print(
        f"detect={trip.dry_run_detection_rate:.2f} "
        f"median_delay={trip.dry_run_median_delay_s:.2f}s "
        f"cv_false={trip.closed_valve_false_trip_rate:.2f} "
        f"healthy_false={trip.healthy_false_trip_rate:.2f}"
    )

    print("\n=== Build demo Twente cache (CI stand-in) ===")
    write_demo_twente_cache(args.demo_twente, n_per_class=8, seed=0)
    records = load_twente(args.demo_twente)
    # Exclude any dry_run if present (should not be)
    records = [r for r in records if r.condition != "dry_run"]

    print(f"\n=== Features profile={args.profile} ===")
    X, y, machines, sources, names = build_feature_table(records, args.profile)
    print(f"X={X.shape} features={len(names)} classes={sorted(set(y))}")

    print("\n=== Confound audit ===")
    report = audit_confound(y.tolist(), machines, sources, X=X)
    print(
        f"NMI class-machine={report.class_machine_nmi:.3f} "
        f"confounded={report.confounded} reasons={report.reasons}"
    )
    if report.warnings:
        print("warnings:", report.warnings)
    assert_not_confounded(report)

    factories = {
        "majority": MajorityClassifier,
        "logistic": lambda: get_baselines()["logistic"],
    }
    try:
        import lightgbm  # noqa: F401

        factories["lightgbm"] = lambda: get_baselines()["lightgbm"]
    except ImportError:
        print("lightgbm not installed; skipping GBDT baseline")

    results = {"_meta": {
        "profile": args.profile,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": names,
        "classes": sorted(set(y.tolist())),
        "machines": sorted(set(machines)),
        "data_source": "twente_demo (SYNTHETIC stand-in, not the real 4TU dataset)",
        "interpretation_caveat": (
            "Scores here are on synthetic signals whose fault signatures were written "
            "into the generator by hand. They verify that the feature pipeline and "
            "splits recover signatures that are known to be present — they are an "
            "upper bound and a wiring check, NOT evidence about real pumps. In "
            "particular the ct_only score is high because the generator encodes clean "
            "torque-modulation sidebands; on real motor current, load variation, "
            "supply distortion and VFD switching noise will degrade it substantially. "
            "No claim in the paper may cite these numbers as a result."
        ),
    }}

    # Both normalisation strategies. The gap between them is a result about how much
    # of the LOMO score depends on seeing the target pump's operating distribution.
    for strategy in NORMALIZATION_STRATEGIES:
        print(f"\n=== LOMO baselines — normalisation={strategy} ===")
        for name, factory in factories.items():
            key = f"{name}__{strategy}"
            results[key] = run_lomo(X, y, machines, factory, name, norm_strategy=strategy)
            r = results[key]
            print(
                f"  {name:10s} macro_f1={r['overall_macro_f1']:.3f} "
                f"acc={r['overall_accuracy']:.3f} "
                f"pr_auc={_fmt(r['per_machine'], 'pr_auc_macro')} "
                f"per_machine={ {k: round(v, 3) for k, v in r['per_machine_macro_f1'].items()} }"
            )

        # McNemar: does the GBDT actually differ from logistic on this test set?
        if "lightgbm" in factories:
            a = results[f"logistic__{strategy}"]
            b = results[f"lightgbm__{strategy}"]
            mc = mcnemar_exact(np.array(a["_y_true"]), np.array(a["_y_pred"]), np.array(b["_y_pred"]))
            results[f"mcnemar_logistic_vs_lightgbm__{strategy}"] = mc
            print(f"  McNemar logistic vs lightgbm: n01={mc['n01']} n10={mc['n10']} p={mc['p_value']:.4f}")

    if not args.skip_tabpfn and tabpfn_available():
        print("\n=== TabPFN v2 ===")
        for strategy in NORMALIZATION_STRATEGIES:
            results[f"tabpfn__{strategy}"] = run_lomo(
                X, y, machines,
                lambda: CachedTabPFN(config=TabPFNConfig(n_estimators=1)),
                "tabpfn_v2",
                norm_strategy=strategy,
            )
            print(f"  {strategy}: {results[f'tabpfn__{strategy}']['overall_macro_f1']:.3f}")
    else:
        print("\n=== TabPFN skipped (not installed or --skip-tabpfn) ===")
        print("    Contributions C2/C4 are UNEVALUATED without this.")

    # Sensor-profile ablation — DESIGN §0.3 requires every experiment to run both.
    other_profile = "ct_only" if args.profile == "full" else "full"
    print(f"\n=== Profile ablation: {other_profile} ===")
    Xc, yc, mc_ids, sc, names_c = build_feature_table(records, other_profile)
    print(f"X={Xc.shape} features={len(names_c)}")
    audit_ct = audit_confound(yc.tolist(), mc_ids, sc, X=Xc)
    assert_not_confounded(audit_ct)
    for name, factory in factories.items():
        key = f"{other_profile}__{name}__unsupervised_per_machine"
        results[key] = run_lomo(
            Xc, yc, mc_ids, factory, f"{name}_{other_profile}",
            norm_strategy="unsupervised_per_machine",
        )
        print(f"  {name:10s} macro_f1={results[key]['overall_macro_f1']:.3f}")

    # Raw predictions are kept in-memory for McNemar but not serialised.
    serialisable = {
        k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            if isinstance(v, dict) else v)
        for k, v in results.items()
    }
    out = args.outdir / f"results_{args.profile}.json"
    out.write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"\nWrote {out}")

    # Sanity check the reader would otherwise have to do by eye. A degenerate run —
    # every model at chance, or all models identical — means a pipeline bug, not a
    # finding, and it must not pass silently into a figure.
    f1s = {
        k: v["overall_macro_f1"]
        for k, v in results.items()
        if isinstance(v, dict) and "overall_macro_f1" in v and not k.startswith("majority")
    }
    maj = [v["overall_macro_f1"] for k, v in results.items()
           if isinstance(v, dict) and k.startswith("majority")]
    if f1s and maj and all(v <= max(maj) + 1e-9 for v in f1s.values()):
        print(
            "\n*** WARNING: no model beats the majority baseline. "
            "This is a pipeline defect signature, not a result. ***"
        )


if __name__ == "__main__":
    main()
