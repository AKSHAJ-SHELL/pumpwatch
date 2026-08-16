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
from pumpwatch.evaluate import bootstrap_ci, mcnemar_exact, recall_at_fixed_far
from pumpwatch.experiment import build_ladder, run_split
from pumpwatch.gateway.baselines import (
    MajorityClassifier,
    get_baselines,
    make_lightgbm,
)
from pumpwatch.gateway.tabpfn_clf import (
    AbstentionConfig,
    CachedTabPFN,
    TabPFNConfig,
    tabpfn_available,
)
from pumpwatch.splits import NORMALIZATION_STRATEGIES


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
    parser.add_argument("--skip-tabpfn", action="store_true")
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

    factories = {
        "majority": MajorityClassifier,
        "logistic": lambda: get_baselines()["logistic"],
    }
    try:
        make_lightgbm()
        factories["lightgbm"] = lambda: get_baselines()["lightgbm"]
    except ImportError:
        print("lightgbm not installed; skipping GBDT baseline")
    if not args.skip_tabpfn and tabpfn_available():
        factories["tabpfn"] = lambda: CachedTabPFN(config=TabPFNConfig(n_estimators=1))
        factories["tabpfn_noabstain"] = lambda: CachedTabPFN(
            config=TabPFNConfig(n_estimators=1),
            abstention=AbstentionConfig(max_prob_threshold=0.0, enable_mahalanobis=False),
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
            "not_applicable": (
                "No current channel: ct_only, MCSA and dry-run cannot be evaluated "
                "here. No waveform: time-domain and envelope features unavailable. "
                "Order-normalised: absolute-frequency bearing analysis unavailable."
            ),
        }
    }

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
                results[key] = run_split(
                    X, y, machines, factory, name, lomo, norm_strategy=strategy
                )
                r = results[key]
                ci = r["macro_f1_bootstrap_ci"]
                print(
                    f"  {name:17s} macro_f1={r['overall_macro_f1']:.3f} "
                    f"acc={r['overall_accuracy']:.3f} "
                    f"cov={r['overall_coverage']:.2f} "
                    f"per-machine CI [{ci['lo']:.3f}, {ci['hi']:.3f}]"
                )
            names_list = list(factories)
            for i, a_name in enumerate(names_list):
                for b_name in names_list[i + 1:]:
                    a, b = results[f"{a_name}__{strategy}"], results[f"{b_name}__{strategy}"]
                    mc = mcnemar_exact(
                        np.array(a["_y_true"]), np.array(a["_y_pred"]), np.array(b["_y_pred"])
                    )
                    results[f"mcnemar_{a_name}_vs_{b_name}__{strategy}"] = mc
                    print(
                        f"    McNemar {a_name} vs {b_name}: "
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
                lambda n=n_ctx: CachedTabPFN(
                    config=TabPFNConfig(n_estimators=1, max_context_rows=n),
                    abstention=AbstentionConfig(
                        max_prob_threshold=0.0, enable_mahalanobis=False
                    ),
                ),
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
