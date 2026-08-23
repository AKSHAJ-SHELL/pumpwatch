#!/usr/bin/env python3
"""External test of the normalisation result on the Paderborn bearing dataset.

The headline finding — that standardising each unit by its own statistics loses badly
to pooling the training units — rests on a single dataset. ESPset is the only public
source with a genuine cross-machine axis for pumps, so a like-for-like replication is
not available. This is the next best thing: independently collected data, a different
laboratory, a different machine type, a different sensor suite, and a held-out unit
that is a physical bearing rather than a pump.

Two design choices keep the comparison honest.

Bearing geometry is passed as None even though the 6203 specification is public.
ESPset carries no bearing geometry, so computing envelope features here and not there
would compare two different feature spaces and attribute the difference to
normalisation. The generic order-domain features are what both datasets share.

One window is taken per acquisition file rather than chopping each four-second
measurement into many. Chopping is precisely the segment-level leakage the protocol
exists to prevent, and using it here to inflate the sample count would undermine the
paper's own argument.
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
warnings.filterwarnings("ignore", message=".*OOD abstention disabled.*")


def build_table(records):
    from pumpwatch.features import FeatureMeta, extract_features, feature_matrix
    from pumpwatch.node.daq import moving_rms

    vecs, y, units = [], [], []
    groups = {"record": [], "component": [], "operating": []}
    for r in records:
        cur_rms = moving_rms(r.current, r.fs) if r.current is not None else None
        meta = FeatureMeta(
            rpm=r.rpm,
            n_vanes=None,          # not a pump; vane-pass features degrade out
            bearing=None,          # see module docstring: kept comparable with ESPset
            rated_current_a=(float(np.percentile(cur_rms, 95)) if cur_rms is not None else None),
            voltage_available=False,
            profile="full",
        )
        vecs.append(extract_features(
            r.vibration, r.fs,
            current_rms=cur_rms, current_waveform=r.current,
            fs_current=r.fs if r.current is not None else None, meta=meta,
        ))
        y.append(r.condition)
        units.append(r.bearing_id)
        groups["record"].append(r.session_id)
        groups["component"].append(r.bearing_id)
        groups["operating"].append(r.operating_point)
    X, names = feature_matrix(vecs)
    return X, np.asarray(y), units, names, groups


def _leave_one_per_class_out(units, y):
    """Hold out one bearing of EVERY class per fold.

    Plain leave-one-bearing-out is degenerate on this dataset and the failure is
    instructive. Each Paderborn bearing carries exactly one damage state, so a
    single-unit holdout produces a test set containing one class. Removing that unit
    also makes its class rarer in training, biasing every model away from predicting
    the only class present in the test set: the majority baseline scores exactly 0.000
    accuracy under both normalisation strategies, and so does logistic regression. That
    measures the fold construction, not the thing under test.

    Holding out one unit of each class per fold restores a multi-class test set while
    still ensuring no physical bearing is ever on both sides of the split, which is the
    property that matters.
    """
    from pumpwatch.splits import SplitResult, SplitFold

    units = np.asarray(units)
    y = np.asarray(y)
    by_class: dict[str, list[str]] = {}
    for cls in sorted(set(y.tolist())):
        by_class[cls] = sorted({u for u, yy in zip(units.tolist(), y.tolist()) if yy == cls})

    n_folds = min(len(v) for v in by_class.values())
    folds = []
    for k in range(n_folds):
        held = {by_class[c][k] for c in by_class}
        test = np.flatnonzero(np.isin(units, list(held)))
        train = np.flatnonzero(~np.isin(units, list(held)))
        folds.append(SplitFold(
            train_idx=train, test_idx=test, held_out=",".join(sorted(held)),
            level=4, context_idx=train,   # TabPFN context is the training set
        ))
    return SplitResult(level=4, folds=folds, verdict="thesis_test",
                       n_machines_train=[len(set(units[f.train_idx].tolist())) for f in folds],
                       n_machines_test=[len(set(units[f.test_idx].tolist())) for f in folds])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "paderborn")
    ap.add_argument("--max-per-condition", type=int, default=5)
    ap.add_argument("--window-s", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "results_paderborn.json")
    args = ap.parse_args()

    from pumpwatch.datasets.paderborn import (
        PADERBORN_CITATION, PADERBORN_DOI, PADERBORN_LICENCE, load_paderborn,
    )
    from pumpwatch.experiment import run_split
    from pumpwatch.models import build_model_zoo

    records = load_paderborn(args.root, max_per_condition=args.max_per_condition,
                             window_s=args.window_s)
    X, y, units, names, groups = build_table(records)
    print(f"\n=== Paderborn: {len(y)} records, {len(set(units))} bearings, "
          f"{X.shape[1]} features ===")
    print(f"    classes: {dict(Counter(y.tolist()))}")
    print(f"    bearings per class: "
          f"{ {c: len({u for u, yy in zip(units, y) if yy == c}) for c in sorted(set(y))} }")

    skew = float(np.mean([
        max(Counter([yy for u, yy in zip(units, y) if u == unit]).values())
        / sum(1 for u in units if u == unit)
        for unit in sorted(set(units))
    ]))
    print(f"    mean dominant-class share per bearing: {skew:.2f}  "
          f"(1.00 = each unit carries exactly one class)")

    zoo = build_model_zoo(include_tabpfn=False, verbose=False)
    models = {k: v for k, v in zoo.items() if k in ("majority", "logistic", "lightgbm")}

    print("\n=== LEAVE-ONE-BEARING-PER-CLASS-OUT, both normalisation strategies ===")
    print(f"  {'model':12}{'per-unit':>11}{'train-pooled':>14}{'gap':>9}")
    out = {"_meta": {
        "dataset": "paderborn", "real_data": True, "doi": PADERBORN_DOI,
        "licence": PADERBORN_LICENCE, "citation": PADERBORN_CITATION,
        "n_records": int(len(y)), "n_units": len(set(units)),
        "n_features": int(X.shape[1]),
        "held_out_unit": "one physical bearing (NOT a machine — see limitations)",
        "mean_dominant_class_share_per_unit": skew,
    }}
    split = _leave_one_per_class_out(units, y)
    for name, factory in models.items():
        per = {}
        for strategy in ("unsupervised_per_machine", "train_pooled"):
            scores = [
                run_split(X, y, units, factory, name, split,
                          norm_strategy=strategy)["overall_macro_f1"]
                for _ in range(args.seeds)
            ]
            per[strategy] = float(np.mean(scores))
        gap = per["train_pooled"] - per["unsupervised_per_machine"]
        out[f"norm__{name}"] = {**per, "gap": gap}
        print(f"  {name:12}{per['unsupervised_per_machine']:>11.3f}"
              f"{per['train_pooled']:>14.3f}{gap:>+9.3f}")

    gaps = [v["gap"] for k, v in out.items()
            if k.startswith("norm__") and not k.endswith("majority")]
    print(f"\n  pooled beats per-unit on {sum(1 for g in gaps if g > 0)}/{len(gaps)} "
          f"non-trivial models; mean gap {np.mean(gaps):+.3f}")

    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    print(f"\n{PADERBORN_CITATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
