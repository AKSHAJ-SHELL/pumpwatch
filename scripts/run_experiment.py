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
from pumpwatch.baseline_lifecycle import commissioning_length
from pumpwatch.node.gates import evaluate_gate, fit_composite_gate, select_gate_features
from pumpwatch.node.trip import evaluate_trip_path
from pumpwatch.physics import BearingGeometry
from pumpwatch.splits import (
    NORMALIZATION_STRATEGIES,
    describe_fold,
    normalize_features,
    split_component_wise,
    split_cross_operating,
    split_lomo,
    split_random_window,
    split_record_wise,
)
from pumpwatch.synth import Condition, generate_dataset


def build_feature_table(records, profile: str):
    vecs, y, machines, sources = [], [], [], []
    # Grouping keys for the leakage ladder, collected alongside the features so a
    # split can never silently fall back to a weaker rung.
    groups: dict[str, list[str]] = {"record": [], "component": [], "operating": []}
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
            rec_id, comp_id, op_id = "", "", ""
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
            rec_id, comp_id, op_id = r.session_id, r.component_id, r.operating_point

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
        groups["record"].append(rec_id)
        groups["component"].append(comp_id)
        groups["operating"].append(op_id)
    X, names = feature_matrix(vecs)
    return X, np.array(y), machines, sources, names, groups


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
    X, y, machines, sources, names, groups = build_feature_table(records, args.profile)
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

    if not args.skip_tabpfn and tabpfn_available():
        factories["tabpfn"] = lambda: CachedTabPFN(config=TabPFNConfig(n_estimators=1))
    else:
        print("\nTabPFN unavailable (not installed or --skip-tabpfn).")
        print("    Contributions C2/C4 are UNEVALUATED without it.")

    # ---- Stage 1: the MCU gate ---------------------------------------------
    # CompositeGate was never instantiated anywhere, so the escalation rate — the
    # quantity that links classification accuracy to battery life, and the only
    # quantitative content of contribution C1 — did not exist.
    # Protocol note: the gate is commissioned PER PUMP on that pump's own healthy
    # baseline, because that is the deployment model — a node is installed on a
    # known-good pump and watches for departures from *its* normal. This is a
    # different protocol from the classifier's LOMO, which asks whether a reference
    # set transfers across pumps. Fitting the gate on other pumps' healthy data
    # instead makes every window on the target pump look anomalous and escalates
    # 100% of them; that is a statement about between-pump variability, not about
    # the gate. Commissioning windows are excluded from evaluation.
    print("\n=== MCU gate (stage 1), commissioned per pump ===")
    gate_results = {}
    rng_gate = np.random.default_rng(0)
    for machine in sorted(set(machines)):
        m_idx = np.flatnonzero(np.asarray(machines) == machine)
        healthy_idx = m_idx[y[m_idx] == "healthy"]
        if len(healthy_idx) < 10:
            print(f"  [skip] {machine}: too few healthy commissioning samples")
            continue
        # Half the healthy windows commission the node; the rest are evaluation.
        shuffled = rng_gate.permutation(healthy_idx)
        n_fit = len(shuffled) // 2
        fit_idx, held_healthy = shuffled[:n_fit], shuffled[n_fit:]
        eval_idx = np.concatenate([held_healthy, m_idx[y[m_idx] != "healthy"]])

        # The gate runs on a small feature subset, not the full vector — its
        # dimensionality is bounded by how long commissioning takes, not by what
        # the extractor can compute.
        gate_cols = select_gate_features(names)
        gate_names = [names[i] for i in gate_cols]
        Xn = normalize_features(X, machines, fit_idx)[:, gate_cols]

        plan = commissioning_length(len(gate_cols))
        adequate = n_fit >= plan.min_samples

        gate = fit_composite_gate(Xn[fit_idx], feature_names=gate_names)
        stats = evaluate_gate(gate, Xn[eval_idx], y[eval_idx])
        stats["n_commissioning"] = int(n_fit)
        stats["n_gate_features"] = len(gate_cols)
        stats["commissioning_required"] = plan.min_samples
        stats["commissioning_adequate"] = bool(adequate)
        if not adequate:
            print(
                f"  [warn] {machine}: {n_fit} healthy windows < {plan.min_samples} "
                f"required for p={len(gate_cols)} — baseline is under-conditioned"
            )
        gate_results[machine] = stats
        print(
            f"  {machine}: escalate healthy={stats['escalation_rate_healthy']:.2f} "
            f"faulty={stats['escalation_rate_faulty']:.2f} "
            f"overall={stats['escalation_rate_overall']:.2f} "
            f"(commissioned on {n_fit} healthy windows) reasons={stats['reasons']}"
        )
    results["gate_stage1"] = gate_results
    if gate_results:
        mean_esc = float(np.mean([g["escalation_rate_overall"] for g in gate_results.values()]))
        mean_field = float(np.mean([g["escalation_rate_field"] for g in gate_results.values()]))
        mean_recall = float(np.mean([g["escalation_rate_faulty"] for g in gate_results.values()]))
        energy = event_triggered_energy(3.0, escalation_rate=mean_field)
        # End-to-end recall is capped by the gate: the gateway never sees what
        # stage 1 suppressed.
        results["gate_summary"] = {
            "mean_escalation_rate_testset": mean_esc,
            "mean_field_escalation_rate": mean_field,
            "gate_recall_ceiling": mean_recall,
            "battery_years_at_field_rate": energy.battery_years,
            "uplinks_per_day_at_field_rate": energy.transmissions_per_day,
            "energy_breakdown_mAh_per_day": energy.breakdown_mAh,
            "tx_fraction": energy.tx_fraction,
            "finding_energy_bottleneck": (
                "With a working gate the radio is NOT the dominant cost: LoRa TX is "
                f"{100 * energy.tx_fraction:.0f}% of the budget while continuous "
                "CUSUM sampling during pump runtime is the bulk of it. This inverts "
                "the v1.0 assumption that TX dominates at ~55%. Because dry-run "
                "protection requires continuous current monitoring whenever the pump "
                "runs (DESIGN §0.2), the optimisation target moves from transmitting "
                "less often to sampling more cheaply — a lower-power comparator or a "
                "duty-cycled CUSUM front end, not a smaller payload."
            ),
            "note": (
                "Gateway classification accuracy is an upper bound conditioned on "
                "escalation: end-to-end fault recall <= gate_recall_ceiling. The "
                "test-set escalation rate reflects how many faulty examples were "
                "collected, not field prevalence; battery life is driven by the "
                "field rate, which is dominated by the healthy false-escalation rate."
            ),
        }
        print(
            f"  mean escalation (test set)={mean_esc:.2f}  "
            f"field-weighted={mean_field:.3f}  recall ceiling={mean_recall:.2f}"
        )
        print(
            f"  -> {energy.transmissions_per_day:.1f} uplinks/day, "
            f"{energy.battery_years:.2f} yr battery at 3 h/day runtime"
        )

    # ---- The leakage ladder -------------------------------------------------
    # Levels 0-3 were implemented, verdict-labelled and never called; only LOMO
    # ever ran. Running the whole ladder is what turns "random splits leak" from
    # an assertion into a measurement.
    print("\n=== Leakage ladder ===")
    ladder = build_ladder(machines, groups, n_samples=X.shape[0])
    for rung, split in ladder.items():
        print(f"\n  --- {rung} ({split.verdict}, {len(split.folds)} folds) ---")
        for name, factory in factories.items():
            key = f"ladder__{rung}__{name}"
            results[key] = run_split(
                X, y, machines, factory, name, split,
                norm_strategy="unsupervised_per_machine",
            )
            print(f"    {name:10s} macro_f1={results[key]['overall_macro_f1']:.3f}")

    # ---- LOMO under both normalisation strategies ---------------------------
    # The gap measures how much of the score depends on seeing the target pump's
    # operating distribution at all.
    lomo = ladder.get("4_lomo")
    if lomo is not None:
        for strategy in NORMALIZATION_STRATEGIES:
            print(f"\n=== LOMO — normalisation={strategy} ===")
            for name, factory in factories.items():
                key = f"{name}__{strategy}"
                results[key] = run_split(
                    X, y, machines, factory, name, lomo, norm_strategy=strategy
                )
                r = results[key]
                print(
                    f"  {name:10s} macro_f1={r['overall_macro_f1']:.3f} "
                    f"acc={r['overall_accuracy']:.3f} "
                    f"pr_auc={_fmt(r['per_machine'], 'pr_auc_macro')} "
                    f"cov={r['overall_coverage']:.2f} "
                    f"per_machine={ {k: round(v, 3) for k, v in r['per_machine_macro_f1'].items()} }"
                )

            # McNemar between every model pair on the same test set.
            pairs = [(a, b) for i, a in enumerate(factories) for b in list(factories)[i + 1:]]
            for a_name, b_name in pairs:
                a, b = results[f"{a_name}__{strategy}"], results[f"{b_name}__{strategy}"]
                mc = mcnemar_exact(
                    np.array(a["_y_true"]), np.array(a["_y_pred"]), np.array(b["_y_pred"])
                )
                results[f"mcnemar_{a_name}_vs_{b_name}__{strategy}"] = mc
                print(
                    f"  McNemar {a_name} vs {b_name}: "
                    f"n01={mc['n01']} n10={mc['n10']} p={mc['p_value']:.4f}"
                )

    # ---- Sensor-profile ablation -------------------------------------------
    # DESIGN §0.3 requires every classification experiment to run both profiles.
    other_profile = "ct_only" if args.profile == "full" else "full"
    print(f"\n=== Profile ablation: {other_profile} ===")
    Xc, yc, mc_ids, sc, names_c, _ = build_feature_table(records, other_profile)
    print(f"X={Xc.shape} features={len(names_c)}")
    assert_not_confounded(audit_confound(yc.tolist(), mc_ids, sc, X=Xc))
    if lomo is not None:
        for name, factory in factories.items():
            key = f"{other_profile}__{name}__unsupervised_per_machine"
            results[key] = run_split(
                Xc, yc, mc_ids, factory, f"{name}_{other_profile}", lomo,
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
