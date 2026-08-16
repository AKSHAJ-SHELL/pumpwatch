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
from pumpwatch.evaluate import classify_report, friedman_nemenyi_allowed, mcnemar_exact
from pumpwatch.features import FeatureMeta, extract_features, feature_matrix
from pumpwatch.gateway.baselines import MajorityClassifier, fit_predict, get_baselines
from pumpwatch.gateway.tabpfn_clf import CachedTabPFN, TabPFNConfig, tabpfn_available
from pumpwatch.node.energy import event_triggered_energy, fixed_schedule_energy
from pumpwatch.node.trip import evaluate_trip_path
from pumpwatch.physics import BearingGeometry
from pumpwatch.splits import describe_fold, normalize_per_machine, split_lomo
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
            rpm = r.rpm
            rated = 10.0
            n_vanes = 6
            bearing = BearingGeometry(8, 7.0, 35.0)
            src = getattr(r, "source", "twente")

        # Skip dry_run in cross-source classifier paths
        meta = FeatureMeta(
            rpm=rpm,
            n_vanes=n_vanes,
            bearing=bearing if profile == "full" else None,
            rated_current_a=rated,
            voltage_available=False,
            profile=profile,
        )
        fv = extract_features(vib, fs, current_rms=current, meta=meta)
        vecs.append(fv)
        y.append(label)
        machines.append(pump_id)
        sources.append(src)
    X, names = feature_matrix(vecs)
    return X, np.array(y), machines, sources, names


def run_lomo(X, y, machines, model_factory, model_name: str) -> dict:
    result = split_lomo(machines)
    per_machine = {}
    all_true, all_pred = [], []
    for fold in result.folds:
        Xn = normalize_per_machine(X, machines, fold.train_idx)
        model = model_factory()
        pred = fit_predict(
            model,
            Xn[fold.context_idx],
            y[fold.context_idx],
            Xn[fold.test_idx],
            model_name,
        )
        report = classify_report(y[fold.test_idx], pred.y_pred, pred.y_proba, pred.classes)
        per_machine[fold.held_out] = report.macro_f1
        all_true.extend(y[fold.test_idx].tolist())
        all_pred.extend(pred.y_pred.tolist())
        print(json.dumps(describe_fold(fold, machines, y.tolist()), indent=2))
    overall = classify_report(np.array(all_true), np.array(all_pred))
    return {
        "model": model_name,
        "per_machine_macro_f1": per_machine,
        "overall_macro_f1": overall.macro_f1,
        "overall_accuracy": overall.accuracy,
        "n_folds": len(result.folds),
        "friedman_allowed": friedman_nemenyi_allowed(len(result.folds)),
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

    print("\n=== LOMO baselines ===")
    results = {}
    factories = {
        "majority": MajorityClassifier,
        "logistic": lambda: get_baselines()["logistic"],
    }
    try:
        import lightgbm  # noqa: F401

        factories["lightgbm"] = lambda: get_baselines()["lightgbm"]
    except ImportError:
        print("lightgbm not installed; skipping GBDT baseline")

    for name, factory in factories.items():
        results[name] = run_lomo(X, y, machines, factory, name)
        print(name, results[name])

    if not args.skip_tabpfn and tabpfn_available():
        print("\n=== TabPFN v2 (optional) ===")

        def tabpfn_factory():
            # Wrap as sklearn-like for fit_predict — use CachedTabPFN manually
            return "_tabpfn_"

        # Manual LOMO for TabPFN
        from pumpwatch.splits import split_lomo as _lomo

        split = _lomo(machines)
        per = {}
        for fold in split.folds:
            Xn = normalize_per_machine(X, machines, fold.train_idx)
            clf = CachedTabPFN(config=TabPFNConfig(n_estimators=1))
            clf.fit_context(Xn[fold.context_idx], y[fold.context_idx])
            pred = clf.predict(Xn[fold.test_idx])
            rep = classify_report(y[fold.test_idx], pred.y_pred, pred.y_proba, pred.classes)
            per[fold.held_out] = rep.macro_f1
        results["tabpfn"] = {"per_machine_macro_f1": per, "mean": float(np.mean(list(per.values())))}
        print("tabpfn:", results["tabpfn"])
    else:
        print("\n=== TabPFN skipped (not installed or --skip-tabpfn) ===")

    # Profile comparison stub on same data
    print("\n=== CT-only ablation ===")
    Xc, yc, mc, sc, _ = build_feature_table(records, "ct_only")
    audit_ct = audit_confound(yc.tolist(), mc, sc, X=Xc)
    assert_not_confounded(audit_ct)
    results["ct_only_logistic"] = run_lomo(
        Xc, yc, mc, lambda: get_baselines()["logistic"], "logistic_ct_only"
    )
    print(results["ct_only_logistic"])

    out = args.outdir / f"results_{args.profile}.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
