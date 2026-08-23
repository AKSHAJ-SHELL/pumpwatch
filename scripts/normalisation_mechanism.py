#!/usr/bin/env python3
"""Test WHY per-machine normalisation loses to train-pooled on cross-machine data.

Observation: on eleven in-service pumps, standardising each machine by its own
statistics costs ~0.24 macro-F1 against pooling the training machines - larger than
the gap between any two models we compare. That is backwards on its face, because the
per-machine strategy uses strictly more information about the test machine.

Hypothesis: per-machine standardisation centres each pump on its own mean. When a
pump's records are dominated by one class - and in service they always are; ten of our
eleven pumps have one class at over 70% - that mean sits inside the dominant class, so
the operation subtracts away the absolute level that separates healthy from faulty. The
strategy destroys the signal precisely in proportion to how imbalanced the machine is.

Correlational support: damage rises with per-machine class skew, r = +0.64 over eleven
pumps. This script tests it causally. If the mechanism is right, then forcing every
machine to be class-balanced - so its mean no longer sits inside one class - should
shrink the gap. If the gap survives balancing, the explanation is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message="Unknown solver options")


def balance_per_machine(y, machines, seed=0, min_per_class=8):
    """Indices where every machine has equal counts of each class it carries.

    Subsamples the dominant classes down rather than upsampling, so no record is
    duplicated and no synthetic point is invented.
    """
    rng = np.random.default_rng(seed)
    y, machines = np.asarray(y), np.asarray(machines)
    keep = []
    for mach in sorted(set(machines.tolist())):
        idx = np.flatnonzero(machines == mach)
        counts = Counter(y[idx].tolist())
        usable = {c: n for c, n in counts.items() if n >= min_per_class}
        if len(usable) < 2:
            continue                      # cannot balance a single-class machine
        n = min(usable.values())
        for cls in usable:
            cls_idx = idx[y[idx] == cls]
            keep.extend(rng.choice(cls_idx, size=n, replace=False).tolist())
    return np.array(sorted(keep))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "normalisation_mechanism.json")
    args = ap.parse_args()

    from pumpwatch.datasets.espset import espset_order_features, load_espset
    from pumpwatch.experiment import run_split
    from pumpwatch.splits import split_lomo
    from pumpwatch.models import build_model_zoo

    data = load_espset(ROOT / "data" / "espset")
    X_all, names = espset_order_features(data)
    y_all = np.asarray(data.labels)
    m_all = np.asarray(data.machine_ids)

    zoo = build_model_zoo(include_tabpfn=False, verbose=False)
    models = {k: v for k, v in zoo.items() if k in ("logistic", "lightgbm")}

    def skew(y, m):
        return float(np.mean([
            max(Counter(y[m == mach].tolist()).values()) / int((m == mach).sum())
            for mach in sorted(set(m.tolist()))
        ]))

    conditions = {}
    for label, idx in (("as-collected", np.arange(len(y_all))),
                       ("class-balanced per machine", balance_per_machine(y_all, m_all))):
        X, y, m = X_all[idx], y_all[idx], m_all[idx]
        conditions[label] = {
            "n": int(len(y)), "n_machines": len(set(m.tolist())),
            "mean_dominant_class_share": skew(y, m), "gaps": {},
        }
        print(f"\n=== {label} ===")
        print(f"  {len(y)} records, {len(set(m.tolist()))} machines, "
              f"mean dominant-class share {skew(y, m):.2f}")
        for name, factory in models.items():
            per_seed = {}
            for strategy in ("unsupervised_per_machine", "train_pooled"):
                scores = []
                for seed in range(args.seeds):
                    r = run_split(X, y, m.tolist(), factory, name,
                                  split_lomo(m.tolist()),
                                  norm_strategy=strategy, verbose=False)
                    scores.append(r["overall_macro_f1"])
                per_seed[strategy] = float(np.mean(scores))
            gap = per_seed["train_pooled"] - per_seed["unsupervised_per_machine"]
            conditions[label]["gaps"][name] = {**per_seed, "gap": gap}
            print(f"  {name:10} per-machine={per_seed['unsupervised_per_machine']:.3f}  "
                  f"pooled={per_seed['train_pooled']:.3f}  gap={gap:+.3f}")

    print("\n=== VERDICT ===")
    for name in models:
        a = conditions["as-collected"]["gaps"][name]["gap"]
        b = conditions["class-balanced per machine"]["gaps"][name]["gap"]
        shrink = (1 - b / a) * 100 if a else float("nan")
        print(f"  {name:10} gap {a:+.3f} -> {b:+.3f} on balanced data "
              f"({shrink:.0f}% smaller)")
    print("\n  If the gap largely closes, class imbalance is the mechanism: per-machine")
    print("  standardisation subtracts a mean that sits inside the dominant class.")
    print("  If it survives, the explanation is wrong and should not be published.")

    args.out.write_text(json.dumps(conditions, indent=2))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
