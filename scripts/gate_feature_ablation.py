#!/usr/bin/env python3
"""Isolate gate feature COUNT from gate feature CHOICE.

The headline comparison in the results - a 5-feature gate reaching a 0.98 recall
ceiling where a 7-feature one reaches 0.83 - is confounded. The two sets differ in
size *and* in origin: the small one is ESPset's own published feature columns, chosen
by domain experts for this fleet, while the large one is what our extractor computes
generically from a spectrum. Attributing the gap to feature count alone would be
wrong, and the design claim it was offered as evidence for (DESIGN 1.3, the gate is
bounded by commissioning length rather than feature count) deserves a real test.

This runs every subset of the deployable order-spectrum features at every size, so
count varies while provenance is held fixed. 2^7 - 1 = 127 subsets across 11 machines,
which is cheap because the gate is a Mahalanobis distance and an EWMA, not a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "gate_feature_ablation.json")
    args = ap.parse_args()

    from pumpwatch.datasets.espset import load_espset, espset_order_features
    from pumpwatch.experiment import run_gate_per_machine, summarise_gate
    from pumpwatch.node.gates import GATE_FEATURE_SETS

    # Only the features our own extractor computes from a spectrum. The published
    # columns are excluded on purpose: they are what makes the original comparison
    # confounded, and a deployed node does not have them.
    data = load_espset(ROOT / "data" / "espset")
    X, names = espset_order_features(data)
    names = list(names)
    y = data.labels
    machines = data.machine_ids.tolist()

    pool = [f for f in GATE_FEATURE_SETS["order_spectrum"] if f in names]
    print(f"Deployable gate features available: {len(pool)}")
    for f in pool:
        print(f"  {f}")

    rows = []
    for k in range(args.min_size, len(pool) + 1):
        for subset in combinations(pool, k):
            cols = [names.index(f) for f in subset]
            sub_names = list(subset)
            res = run_gate_per_machine(
                X[:, cols], y, machines, sub_names, verbose=False
            )
            if not res:
                continue
            s = summarise_gate(res)
            rows.append({
                "k": k,
                "features": sub_names,
                "field_escalation": s["mean_field_escalation_rate"],
                "ceiling_mean": s["gate_recall_ceiling"],
                "ceiling_worst": s["gate_recall_ceiling_worst_machine"],
                "n_commissioned": s["n_machines_adequately_commissioned"],
                "n_machines": s["n_machines"],
            })
        print(f"  k={k}: {sum(1 for r in rows if r['k'] == k)} subsets evaluated")

    print(f"\n{'k':>3} {'subsets':>8} {'best ceiling':>13} {'median':>9} {'worst-pump(best)':>17} {'commissioned':>13}")
    summary = {}
    for k in sorted({r["k"] for r in rows}):
        ks = [r for r in rows if r["k"] == k]
        ceilings = sorted(r["ceiling_mean"] for r in ks)
        best = max(ks, key=lambda r: r["ceiling_mean"])
        summary[k] = {
            "n_subsets": len(ks),
            "best_ceiling": best["ceiling_mean"],
            "median_ceiling": float(np.median(ceilings)),
            "best_features": best["features"],
            "best_worst_pump": best["ceiling_worst"],
            "best_field_escalation": best["field_escalation"],
            "commissioned_at_best": f"{best['n_commissioned']}/{best['n_machines']}",
        }
        print(
            f"{k:>3} {len(ks):>8} {best['ceiling_mean']:>13.3f} "
            f"{np.median(ceilings):>9.3f} {best['ceiling_worst']:>17.3f} "
            f"{best['n_commissioned']:>8}/{best['n_machines']}"
        )

    print("\nBest subset at each size:")
    for k, v in summary.items():
        print(f"  k={k}: {', '.join(v['best_features'])}")

    args.out.write_text(json.dumps({"per_subset": rows, "by_size": summary}, indent=2))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
