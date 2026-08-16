#!/usr/bin/env python3
"""Twente/4TU experiment on the REAL download.

What this dataset can and cannot support, established by inspection rather than
assumption (see datasets/twente_raw):

* **Cross-operating-condition splits are real here.** Motor-2 was run at 50%, 75%
  and 100% speed, so holding out a speed tests the thing a VFD-driven pump
  actually does. ESPset cannot do this; its spectra are order-normalised.
* **Leave-one-machine-out is impossible.** The two motors carry disjoint fault
  sets — the only labels they share are the healthy variants. This script checks
  and refuses rather than reporting a meaningless number.
* **Both vibration and current are present**, which makes this the only real data
  in the project that can compare the `full` and `ct_only` profiles.

Pairing caveat: vibration and current bursts were acquired on separate schedules,
so attaching burst *i* of one to burst *i* of the other is an approximation
justified only by each folder being a steady-state run. It is required for the
profile comparison, is enabled explicitly, and is recorded in the results.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pumpwatch.audit import audit_confound
from pumpwatch.datasets.twente_raw import (
    TWENTE_RAW_DOI,
    load_twente_raw,
    lomo_feasible,
)
from pumpwatch.evaluate import mcnemar_exact
from pumpwatch.experiment import run_split
from pumpwatch.features import FeatureMeta, extract_features, feature_matrix
from pumpwatch.gateway.baselines import MajorityClassifier, get_baselines, make_lightgbm
from pumpwatch.gateway.tabpfn_clf import (
    AbstentionConfig,
    CachedTabPFN,
    TabPFNConfig,
    tabpfn_available,
)
from pumpwatch.splits import (
    split_component_wise,
    split_cross_operating,
    split_random_window,
    split_label_coverage,
    split_record_wise,
)


def moving_rms(x: np.ndarray, fs: float, win_s: float = 0.02) -> np.ndarray:
    """RMS envelope of a current waveform.

    The dataset stores raw current, but the trip path and the current-level
    features are defined on an RMS trajectory, so it is derived here rather than
    the waveform being passed where an envelope is expected.
    """
    n = max(int(win_s * fs), 1)
    kernel = np.ones(n) / n
    return np.sqrt(np.convolve(np.asarray(x, dtype=float) ** 2, kernel, mode="same"))


def build_table(records, profile: str):
    vecs, y, machines = [], [], []
    groups = {"record": [], "component": [], "operating": []}
    for r in records:
        cur_rms = moving_rms(r.current, r.fs) if r.current is not None else None
        meta = FeatureMeta(
            rpm=r.rpm,
            n_vanes=r.n_vanes,  # None for this dataset — VPF features degrade out
            bearing=None,  # bearing geometry not published in machine-readable form
            rated_current_a=(float(np.percentile(cur_rms, 95)) if cur_rms is not None else None),
            voltage_available=False,
            profile=profile,
        )
        vecs.append(
            extract_features(
                r.vibration,
                r.fs,
                current_rms=cur_rms,
                current_waveform=r.current,
                fs_current=r.fs if r.current is not None else None,
                meta=meta,
            )
        )
        y.append(r.condition)
        machines.append(r.pump_id)
        groups["record"].append(r.session_id)
        groups["component"].append(r.component_id)
        groups["operating"].append(r.operating_point)
    X, names = feature_matrix(vecs)
    return X, np.array(y), machines, names, groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "twente_sel")
    p.add_argument("--max-bursts", type=int, default=8)
    p.add_argument("--window-s", type=float, default=2.0)
    p.add_argument("--skip-tabpfn", action="store_true")
    p.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not (args.root / "Vibration").exists():
        print(f"No extracted Twente tree at {args.root}. See datasets/twente_raw.")
        return 1

    print("=== Twente/4TU (REAL data) ===")
    records = load_twente_raw(
        args.root,
        max_bursts=args.max_bursts,
        window_s=args.window_s,
        pair_channels=True,
    )
    print(f"records: {len(records)}")

    feas = lomo_feasible(records)
    print("\n=== Is leave-one-machine-out possible here? ===")
    print(json.dumps(feas, indent=2))
    if not feas["lomo_feasible"]:
        print(
            "\n*** LOMO REFUSED on Twente: the motors share no fault class, so every\n"
            "    fold would train and test on disjoint label sets. Cross-machine\n"
            "    claims must come from ESPset. ***"
        )

    X, y, machines, names, groups = build_table(records, "full")
    print(f"\nfull profile: X={X.shape} classes={sorted(set(y.tolist()))}")

    print("\n=== Confound audit (this is the merge the audit exists to catch) ===")
    rep = audit_confound(y.tolist(), machines, ["twente"] * len(y), X=X)
    print(f"class-machine NMI={rep.class_machine_nmi:.3f} confounded={rep.confounded}")
    for r in rep.reasons:
        print("  REASON:", r)

    factories = {
        "majority": MajorityClassifier,
        "logistic": lambda: get_baselines()["logistic"],
    }
    try:
        make_lightgbm()
        factories["lightgbm"] = lambda: get_baselines()["lightgbm"]
    except ImportError:
        pass
    if not args.skip_tabpfn and tabpfn_available():
        factories["tabpfn"] = lambda: CachedTabPFN(
            config=TabPFNConfig(n_estimators=1),
            abstention=AbstentionConfig(max_prob_threshold=0.0, enable_mahalanobis=False),
        )

    results = {
        "_meta": {
            "dataset": "twente_4tu",
            "real_data": True,
            "doi": TWENTE_RAW_DOI,
            "licence": "CC BY 4.0",
            "n_records": len(records),
            "n_features": int(X.shape[1]),
            "feature_names": names,
            "classes": sorted(set(y.tolist())),
            "max_bursts": args.max_bursts,
            "window_s": args.window_s,
            "lomo_feasibility": feas,
            "channel_pairing_caveat": (
                "Vibration and current bursts were acquired on separate schedules; "
                "burst i of one is paired with burst i of the other. Defensible only "
                "because each folder is a steady-state run of one condition."
            ),
            "unavailable_features": (
                "Impeller vane counts and bearing geometry are not published in "
                "machine-readable form, so VPF, VPF-sideband and bearing-envelope "
                "features degrade out rather than being computed at invented "
                "frequencies."
            ),
        }
    }

    # Only the rungs this dataset can actually support.
    ladder = {
        "0_random_window": split_random_window(len(y), seed=0),
        "1_record_wise": split_record_wise(groups["record"]),
        "2_component_wise": split_component_wise(groups["component"]),
        "3_cross_operating": split_cross_operating(groups["operating"]),
    }
    print("\n=== Leakage ladder (real data; LOMO omitted as infeasible) ===")
    coverage = {rung: split_label_coverage(s, y.tolist()) for rung, s in ladder.items()}
    results["_meta"]["split_label_coverage"] = coverage
    for rung, cov in coverage.items():
        if not cov["interpretable"]:
            print(
                f"  [!] {rung}: {cov['fraction_test_classes_unseen']:.0%} of test "
                f"classes are absent from their own training fold — this rung's "
                f"score reflects the split's degeneracy, not the model"
            )

    for rung, split in ladder.items():
        cov = coverage[rung]
        flag = "" if cov["interpretable"] else "  [NOT INTERPRETABLE]"
        print(f"\n  --- {rung} ({split.verdict}, {len(split.folds)} folds){flag} ---")
        for name, factory in factories.items():
            key = f"ladder__{rung}__{name}"
            results[key] = run_split(
                X, y, machines, factory, name, split,
                norm_strategy="train_pooled",
            )
            print(
                f"    {name:10s} macro_f1={results[key]['overall_macro_f1']:.3f} "
                f"acc={results[key]['overall_accuracy']:.3f}"
            )

    # Sensor-profile comparison on real data — the only place this is possible.
    print("\n=== Profile comparison on REAL data (cross-operating split) ===")
    Xc, yc, mc, names_c, groups_c = build_table(records, "ct_only")
    print(f"ct_only: X={Xc.shape}")
    results["_meta"]["n_features_ct_only"] = int(Xc.shape[1])
    split_c = split_cross_operating(groups_c["operating"])
    for name, factory in factories.items():
        key = f"ct_only__{name}"
        results[key] = run_split(
            Xc, yc, mc, factory, f"{name}_ct_only", split_c, norm_strategy="train_pooled"
        )
        full_key = f"ladder__3_cross_operating__{name}"
        print(
            f"  {name:10s} full={results[full_key]['overall_macro_f1']:.3f}  "
            f"ct_only={results[key]['overall_macro_f1']:.3f}"
        )

    # Cross-operating, restricted to the one motor that was run at several speeds.
    # Across both motors the rung is degenerate: Motor-4 ran only at 70%, so holding
    # that operating point out removes cavitation, misalignment and unbalance
    # entirely. Restricted to Motor-2 the question is the real one — does a model
    # trained at 50% and 75% speed still work at 100%?
    m2 = np.array([m == "NK80-250" for m in machines])
    if m2.sum() > 0:
        y2 = y[m2]
        ops2 = [o for o, k in zip(groups["operating"], m2) if k]
        if len(set(ops2)) >= 2:
            split_m2 = split_cross_operating(ops2)
            cov2 = split_label_coverage(split_m2, y2.tolist())
            results["_meta"]["cross_operating_motor2_coverage"] = cov2
            print(
                f"\n=== Cross-operating within Motor-2 only "
                f"({len(split_m2.folds)} speeds, interpretable={cov2['interpretable']}) ==="
            )
            m2_machines = [m for m, k in zip(machines, m2) if k]
            for name, factory in factories.items():
                key = f"cross_operating_motor2__{name}"
                results[key] = run_split(
                    X[m2], y2, m2_machines, factory, name, split_m2,
                    norm_strategy="train_pooled",
                )
                print(
                    f"  {name:10s} macro_f1={results[key]['overall_macro_f1']:.3f} "
                    f"acc={results[key]['overall_accuracy']:.3f}"
                )

    # Headline contrast uses record-wise, the strongest rung that is interpretable
    # on this subset — comparing against a degenerate rung would overstate it.
    rand = results.get("ladder__0_random_window__lightgbm")
    rec = results.get("ladder__1_record_wise__lightgbm")
    if rand and rec and coverage["1_record_wise"]["interpretable"]:
        results["leakage_inflation"] = {
            "random_window_macro_f1": rand["overall_macro_f1"],
            "record_wise_macro_f1": rec["overall_macro_f1"],
            "inflation_factor": rand["overall_macro_f1"] / max(rec["overall_macro_f1"], 1e-9),
            "compared_rung": "1_record_wise",
            "note": (
                "Both rungs are interpretable (every fold trains on the classes it "
                "tests). The gap is what holding out the recording session costs — "
                "i.e. how much of the random-split score was memorised setup."
            ),
        }
        print(
            f"\n*** Leakage inflation (lightgbm): random-window "
            f"{rand['overall_macro_f1']:.3f} vs record-wise "
            f"{rec['overall_macro_f1']:.3f} "
            f"({results['leakage_inflation']['inflation_factor']:.1f}x) ***"
        )

    serialisable = {
        k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            if isinstance(v, dict) else v)
        for k, v in results.items()
    }
    serialisable["_meta"] = results["_meta"]
    out = args.outdir / "results_twente_real.json"
    out.write_text(json.dumps(serialisable, indent=2, default=str))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
