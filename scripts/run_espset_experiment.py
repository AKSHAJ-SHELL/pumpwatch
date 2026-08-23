#!/usr/bin/env python3
"""ESPset experiment — the leakage ladder and LOMO on REAL multi-machine data.

This is the only experiment in the project that runs on real in-service machines,
and it is where the methodological claims get tested rather than rehearsed:

* 11 distinct pumps, so leave-one-machine-out has 11 folds instead of the demo
  cache's 2. Machine-level bootstrap CIs and the >=5-dataset Friedman guard finally
  have enough units to mean something.
* Field class prevalence (~84% healthy) rather than a balanced rig, which is why
  accuracy is close to useless here and macro-F1 / PR-AUC are the headline.
* Faults as they occur, not as they are seeded.

Constraints inherited from the data (see datasets/espset): spectra only, so no
time-domain or envelope features; order-normalised, so no absolute-frequency
bearing analysis; vibration only, so nothing here touches the ct_only profile,
MCSA, or dry running.
"""

from __future__ import annotations

import os

# Must precede any OpenMP-loading import — see scripts/run_experiment.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings

# sklearn 1.6 passes `iprint` to scipy's lbfgs, which newer scipy rejects. It is a
# verbosity flag with no effect on the fit, but it is emitted once per
# LogisticRegression call — hundreds of lines across 11 folds x 5 seeds x a tuning
# grid, which buries the actual results. Silenced narrowly by message so a real
# convergence warning still gets through.
warnings.filterwarnings("ignore", message=".*Unknown solver options: iprint.*")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pumpwatch.audit import audit_confound
from pumpwatch.datasets.espset import (
    ESPSET_CITATION,
    ESPSET_LICENCE,
    espset_available,
    espset_order_features,
    load_espset,
)
from pumpwatch.evaluate import (
    classify_report,
    mcnemar_exact,
    recall_at_alarm_budget,
)
from pumpwatch.experiment import (
    build_ladder,
    run_gate_per_machine,
    run_split,
    run_split_repeated,
    summarise_gate,
)
from pumpwatch.gateway.baselines import (
    fit_predict,
    make_lightgbm,
    make_logistic,
)
from pumpwatch.models import TABPFN_NOABSTAIN, build_model_zoo, model_pairs
from pumpwatch.tuning import DEFAULT_GRIDS, tuned_factory
from pumpwatch.gateway.tabpfn_clf import (
    tabpfn_available,
)
from pumpwatch.duty import DEFAULT_DUTY, duty_for_decisions_per_month
from pumpwatch.splits import NORMALIZATION_STRATEGIES, normalize_features


def build_espset_table(root: Path, feature_set: str, drop_sensor_faults: bool):
    data = load_espset(root, drop_sensor_faults=drop_sensor_faults)
    mine, mine_names = espset_order_features(data)

    if feature_set == "order":
        X, names = mine, mine_names
    elif feature_set == "published":
        if data.published_features is None:
            raise ValueError("features.csv did not carry the published feature columns")
        X, names = data.published_features, list(data.published_feature_names)
    elif feature_set == "both":
        X = np.hstack([mine, data.published_features])
        names = mine_names + list(data.published_feature_names)
    else:
        raise ValueError(f"unknown feature_set {feature_set!r}")

    machines = data.machine_ids.tolist()
    # ESPset has no session or component metadata, so record-wise and
    # component-wise rungs are genuinely unavailable rather than approximated.
    groups = {
        "record": [""] * len(machines),
        "component": [""] * len(machines),
        "operating": [""] * len(machines),
    }
    return X, data.labels, machines, names, groups, data


def _alarm_budget_table(X, y, machines, lomo, factories, duty=None) -> dict:
    """Fault recall at the alarm budget, pooled across LOMO folds.

    Needs probabilities, so it re-runs the folds rather than reusing the stored
    per-fold summaries — the harness keeps reports, not raw predicted scores.
    """
    out = {}
    for name, factory in factories.items():
        true_all, score_all, classes = [], [], None
        for fold in lomo.folds:
            Xn = normalize_features(X, machines, fold.train_idx, strategy="train_pooled")
            p = fit_predict(
                factory(), Xn[fold.context_idx], y[fold.context_idx],
                Xn[fold.test_idx], name,
            )
            if p.y_proba is None or p.classes is None:
                break
            classes = p.classes
            true_all.extend(y[fold.test_idx].tolist())
            score_all.append(np.asarray(p.y_proba, dtype=float))
        if classes is None or not score_all:
            out[name] = None
            continue
        y_arr, proba = np.array(true_all), np.vstack(score_all)
        out[name] = recall_at_alarm_budget(
            y_arr, proba, classes, **_duty_kwargs(duty or DEFAULT_DUTY)
        )
        # The whole cadence sweep, not only the chosen point. The operating point is a
        # design parameter worth more than the model choice, so a results file that
        # records one point cannot support the claim; the paper's table is generated
        # from this rather than hand-copied.
        out[name]["cadence_sweep"] = {
            str(int(n)): recall_at_alarm_budget(
                y_arr, proba, classes,
                **_duty_kwargs(duty_for_decisions_per_month(n)),
            )
            for n in (1080, 360, 90, 30, 12)
        }
    return out


def _duty_kwargs(d) -> dict:
    """Duty cycle -> the keyword arguments evaluate's helpers take."""
    return {
        "windows_per_runtime_hour": d.decision_windows_per_runtime_hour,
        "runtime_hours_per_day": d.runtime_hours_per_day,
        "days": d.days_per_month,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "espset")
    parser.add_argument(
        "--feature-set", choices=["order", "published", "both"], default="both"
    )
    parser.add_argument(
        "--keep-sensor-faults",
        action="store_true",
        help="Keep the 'faulty_sensor' class. It is an instrumentation problem, not "
        "a machine condition, so it is dropped by default.",
    )
    parser.add_argument(
        "--decisions-per-month",
        type=float,
        default=None,
        help="Gateway decisions per pump-month, which sets the false-alarm budget for "
        "the one-alarm-per-month promise. Defaults to the shipped duty cycle "
        "(30/month, one per runtime day). The original 1080 was what capped "
        "end-to-end recall at 0.086; see pumpwatch.duty.",
    )
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Repeat each split under N seeds and report mean +/- spread. "
             "TabPFN randomises its ensemble permutations, so a single run is one "
             "draw from a distribution nobody has measured.",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Nested, machine-grouped hyperparameter search for the baselines.",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not espset_available(args.root):
        print(f"ESPset not found at {args.root}. See datasets/espset for download.")
        return 1

    X, y, machines, names, groups, data = build_espset_table(
        args.root, args.feature_set, drop_sensor_faults=not args.keep_sensor_faults
    )
    print("=== ESPset (REAL field data) ===")
    print(json.dumps(data.describe(), indent=2))
    print(f"\nfeature_set={args.feature_set}  X={X.shape}")
    print(f"classes={sorted(set(y.tolist()))}")
    print(f"machines={len(set(machines))}")

    print("\n=== Confound audit ===")
    rep = audit_confound(y.tolist(), machines, ["espset"] * len(y), X=X)
    print(f"class-machine NMI={rep.class_machine_nmi:.3f} confounded={rep.confounded}")
    for r in rep.reasons:
        print("  REASON:", r)
    for w in rep.warnings:
        print("  WARN:", w)

    factories = build_model_zoo(include_tabpfn=not args.skip_tabpfn)

    duty = (
        duty_for_decisions_per_month(args.decisions_per_month)
        if args.decisions_per_month
        else DEFAULT_DUTY
    )

    results = {
        "_meta": {
            "dataset": "espset",
            "real_data": True,
            "doi": "10.17632/m268jsw339.3",
            "licence": ESPSET_LICENCE,
            "citation": ESPSET_CITATION,
            "feature_set": args.feature_set,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_names": names,
            "n_machines": len(set(machines)),
            "classes": sorted(set(y.tolist())),
            "modality": "order-normalised velocity spectra (mm/s), vibration only",
            # A recall number without its decision cadence is not interpretable, for
            # the same reason a cross-machine score without its normalisation strategy
            # is not: the cadence sets the false-alarm budget the recall was measured
            # against, and it moves that recall more than the model does.
            "duty_cycle": {
                "decisions_per_month": duty.decisions_per_month,
                "decisions_per_day": duty.decisions_per_day,
                "hours_between_decisions": duty.hours_between_decisions,
                "far_budget_at_1_alarm_per_month": duty.far_for_alarms_per_month(1.0),
                "commissioning_windows_per_runtime_hour": (
                    duty.commissioning_windows_per_runtime_hour
                ),
                "note": (
                    "The one-alarm-per-pump-per-month promise is invariant across "
                    "cadence; only the per-decision specificity required to keep it "
                    "changes. Commissioning cadence is deliberately separate so a "
                    "slower operational cadence never lengthens time-to-usable."
                ),
            },
            "not_applicable": (
                "No current channel: ct_only, MCSA and dry-run cannot be evaluated "
                "here. No waveform: time-domain and envelope features unavailable. "
                "Order-normalised: absolute-frequency bearing analysis unavailable."
            ),
        }
    }

    # The MCU gate on REAL machines. ESPset has 4801 healthy records across 11
    # pumps, which is the commissioning volume the synthetic demo cache cannot
    # reach — so this is where the gate's escalation rate stops being a caveat.
    print("\n=== MCU gate (stage 1), commissioned per pump ===")
    gate_results = run_gate_per_machine(X, y, machines, names)
    results["gate_stage1"] = gate_results
    gate_summary = summarise_gate(gate_results)
    if gate_summary:
        results["gate_summary"] = gate_summary
        print(
            f"  field-weighted escalation={gate_summary['mean_field_escalation_rate']:.3f}  "
            f"recall ceiling={gate_summary['gate_recall_ceiling']:.2f}  "
            f"adequately commissioned on "
            f"{gate_summary['n_machines_adequately_commissioned']}/"
            f"{gate_summary['n_machines']} machines"
        )
        print(
            f"  -> {gate_summary['uplinks_per_day_at_field_rate']:.1f} uplinks/day, "
            f"{gate_summary['battery_years_at_field_rate']:.2f} yr battery"
        )

    print("\n=== Leakage ladder (real machines) ===")
    ladder = build_ladder(machines, groups, n_samples=X.shape[0])
    for rung, split in ladder.items():
        print(f"\n  --- {rung} ({split.verdict}, {len(split.folds)} folds) ---")
        for name, factory in factories.items():
            key = f"ladder__{rung}__{name}"
            results[key] = run_split(
                X, y, machines, factory, name, split,
                norm_strategy="unsupervised_per_machine",
            )
            r = results[key]
            print(
                f"    {name:17s} macro_f1={r['overall_macro_f1']:.3f} "
                f"acc={r['overall_accuracy']:.3f}"
            )

    lomo = ladder.get("4_lomo")
    if lomo is not None:
        for strategy in NORMALIZATION_STRATEGIES:
            print(f"\n=== LOMO — normalisation={strategy} ===")
            for name, factory in factories.items():
                key = f"{name}__{strategy}"
                runner = run_split_repeated if args.seeds > 1 else run_split
                kw = {"seeds": tuple(range(args.seeds))} if args.seeds > 1 else {}
                results[key] = runner(
                    X, y, machines, factory, name, lomo, norm_strategy=strategy, **kw
                )
                r = results[key]
                ci = r["macro_f1_bootstrap_ci"]
                spread = r.get("macro_f1_over_seeds")
                seed_txt = f" ±{spread['std']:.3f}" if spread else ""
                print(
                    f"  {name:17s} macro_f1={r['overall_macro_f1']:.3f}{seed_txt} "
                    f"acc={r['overall_accuracy']:.3f} "
                    f"cov={r['overall_coverage']:.2f} "
                    f"per-machine CI [{ci['lo']:.3f}, {ci['hi']:.3f}]"
                )

        # Tuned baselines. TabPFN has essentially nothing to tune, so comparing it
        # to library-default baselines is not a fair fight — this closes the
        # "you didn't tune the baseline" objection to the headline claim.
        if args.tune:
            print("\n=== Tuned baselines (nested, machine-grouped inner folds) ===")
            tuned = {}
            for name, maker, grid in [
                ("logistic", make_logistic, DEFAULT_GRIDS["logistic"]),
                ("lightgbm", make_lightgbm, DEFAULT_GRIDS["lightgbm"]),
            ]:
                if name not in factories:
                    continue
                per_fold_params = []
                preds_true, preds_pred = [], []
                for fold in lomo.folds:
                    f = tuned_factory(
                        maker, grid,
                        X=X, y=y, machines=machines, train_idx=fold.train_idx,
                        norm_strategy="train_pooled",
                        held_out_machine=fold.held_out,
                    )
                    per_fold_params.append(
                        {"held_out": fold.held_out, **f.tuning_result.best_params}
                    )
                    Xn = normalize_features(
                        X, machines, fold.train_idx, strategy="train_pooled"
                    )
                    p = fit_predict(
                        f(), Xn[fold.context_idx], y[fold.context_idx],
                        Xn[fold.test_idx], name,
                    )
                    preds_true.extend(y[fold.test_idx].tolist())
                    preds_pred.extend(p.y_pred.tolist())
                rep = classify_report(np.array(preds_true), np.array(preds_pred))
                tuned[name] = {
                    "model": f"{name}_tuned",
                    "overall_macro_f1": rep.macro_f1,
                    "overall_accuracy": rep.accuracy,
                    "per_fold_best_params": per_fold_params,
                }
                untuned = results.get(f"{name}__train_pooled", {}).get("overall_macro_f1")
                print(
                    f"  {name:10s} tuned={rep.macro_f1:.3f}"
                    + (f"  (untuned {untuned:.3f})" if untuned is not None else "")
                )
            results["tuned_baselines"] = tuned

        # Recall at the farmer-facing alarm budget, on the best-covered model.
        print("\n=== Recall at ≤1 false alarm / pump / month ===")
        print(f"    operating point: {duty.describe()}")
        results["recall_at_alarm_budget"] = _alarm_budget_table(
            X, y, machines, lomo, factories, duty=duty
        )
        for name, val in results["recall_at_alarm_budget"].items():
            if val:
                print(
                    f"  {name:17s} recall={val['recall']:.3f} at FAR={val['far']:.5f} "
                    f"(budget {val['max_far']:.5f})"
                )
        # Dedented deliberately: this is the pairwise matrix over all models, not a
        # per-model quantity. Nested inside the alarm-budget loop above it printed the
        # entire matrix once per model row - forty identical lines that buried the
        # numbers the section exists to show.
        print("\n=== McNemar, pairwise (exact binomial; Dietterich 1998) ===")
        for a_name, b_name in model_pairs(factories):
            a, b = results[f"{a_name}__{strategy}"], results[f"{b_name}__{strategy}"]
            mc = mcnemar_exact(
                np.array(a["_y_true"]), np.array(a["_y_pred"]), np.array(b["_y_pred"])
            )
            results[f"mcnemar_{a_name}_vs_{b_name}__{strategy}"] = mc
            print(
                f"  {a_name} vs {b_name}: "
                f"n01={mc['n01']} n10={mc['n10']} p={mc['p_value']:.4f}"
            )

        # The headline comparison: an invalid split against the honest one.
        rand = results.get("ladder__0_random_window__lightgbm")
        lomo_r = results.get("lightgbm__unsupervised_per_machine")
        if rand and lomo_r:
            results["leakage_inflation"] = {
                "random_window_macro_f1": rand["overall_macro_f1"],
                "lomo_macro_f1": lomo_r["overall_macro_f1"],
                "inflation_factor": rand["overall_macro_f1"]
                / max(lomo_r["overall_macro_f1"], 1e-9),
                "note": (
                    "Same data, same model, two split protocols. The random-window "
                    "number is what a paper reports when it does not hold out the "
                    "machine; the LOMO number is what the system would actually do "
                    "on a pump it has never seen."
                ),
            }
            print(
                f"\n*** Leakage inflation (lightgbm): "
                f"random-window {rand['overall_macro_f1']:.3f} vs "
                f"LOMO {lomo_r['overall_macro_f1']:.3f} "
                f"({results['leakage_inflation']['inflation_factor']:.2f}x) ***"
            )

    # Context-size sweep: how large does the in-context reference set need to be?
    # This is the operational form of C2 — commissioning a new pump means labelling
    # some windows, and the answer determines how many.
    if not args.skip_tabpfn and tabpfn_available() and lomo is not None:
        print("\n=== TabPFN context-size sweep (LOMO) ===")
        sweep = []
        for n_ctx in [50, 100, 250, 500, 1000]:
            r = run_split(
                X, y, machines,
                build_model_zoo(tabpfn_context_rows=n_ctx, verbose=False)[TABPFN_NOABSTAIN],
                f"tabpfn_ctx{n_ctx}", lomo, norm_strategy="train_pooled",
            )
            sweep.append({
                "n_context": n_ctx,
                "macro_f1": r["overall_macro_f1"],
                "accuracy": r["overall_accuracy"],
                "latency_predict_s": r["mean_latency_predict_s"],
            })
            print(
                f"  ctx={n_ctx:5d}  macro_f1={r['overall_macro_f1']:.3f}  "
                f"predict={r['mean_latency_predict_s']:.2f}s"
            )
        results["tabpfn_context_sweep"] = sweep

    serialisable = {
        k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            if isinstance(v, dict) else v)
        for k, v in results.items()
    }
    serialisable["_meta"] = results["_meta"]
    out = args.outdir / f"results_espset_{args.feature_set}.json"
    out.write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
