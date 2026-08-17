#!/usr/bin/env python3
"""Cross-dataset figures — the ones that need more than one results file.

`make_figures.py` renders a single experiment. These two compare across them,
which is where the strongest claims live: the leakage effect grows as the data
gets more real, and the PCA panels show *why*.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pumpwatch.figures import fig_leakage_across_datasets, fig_pca_class_vs_machine


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def leakage_entries(results_dir: Path) -> list[dict]:
    """Pull (invalid split, honest split) pairs from whichever results exist."""
    entries = []

    synth = _load(results_dir / "results_full.json")
    if synth:
        rand = synth.get("ladder__0_random_window__lightgbm", {})
        lomo = synth.get("lightgbm__unsupervised_per_machine", {})
        if rand and lomo:
            entries.append({
                "dataset": "synthetic",
                "invalid": rand["overall_macro_f1"],
                "honest": lomo["overall_macro_f1"],
                "honest_label": "LOMO, 2 machines",
            })

    for name, label in [
        ("results_espset_both.json", "LOMO, 11 real machines"),
        ("results_espset_published.json", "LOMO, 11 real machines"),
    ]:
        esp = _load(results_dir / name)
        if esp:
            inf = esp.get("leakage_inflation")
            if inf:
                entries.append({
                    "dataset": "ESPset (real)",
                    "invalid": inf["random_window_macro_f1"],
                    "honest": inf["lomo_macro_f1"],
                    "honest_label": label,
                })
            break

    tw = _load(results_dir / "results_twente_real.json")
    if tw:
        inf = tw.get("leakage_inflation")
        if inf:
            entries.append({
                "dataset": "Twente (real)",
                "invalid": inf["random_window_macro_f1"],
                "honest": inf["record_wise_macro_f1"],
                "honest_label": "record-wise",
            })
    return entries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--outdir", type=Path, default=ROOT / "figures" / "summary")
    p.add_argument("--espset-root", type=Path, default=ROOT / "data" / "espset")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    made = []

    entries = leakage_entries(args.results_dir)
    if entries:
        made.append(
            fig_leakage_across_datasets(args.outdir / "D13_leakage_across_datasets.png", entries)
        )
    else:
        print("no leakage_inflation entries found — run the experiments first")

    # B6 on real data: the visual form of the leakage argument.
    from pumpwatch.datasets.espset import espset_available, espset_order_features, load_espset

    if espset_available(args.espset_root):
        d = load_espset(args.espset_root, drop_sensor_faults=True)
        X, _ = espset_order_features(d)
        made.append(
            fig_pca_class_vs_machine(
                args.outdir / "B6_pca_class_vs_machine.png", X, d.labels, d.machine_ids
            )
        )
    else:
        print(f"ESPset not at {args.espset_root}; skipping B6")

    for m in made:
        print(m)


if __name__ == "__main__":
    raise SystemExit(main())
