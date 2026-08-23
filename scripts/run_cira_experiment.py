#!/usr/bin/env python3
"""False-alarm behaviour of the gate on real industrial pumps, with persistence.

The operating-point argument concedes two things it cannot test on ESPset. First, the
cadence result is a deployment counterfactual: we reason about what one decision per
runtime day implies rather than observing a node run at that cadence. Second, a
persistence rule -- alarm only when a fault shows in k of the last n windows -- ought
to beat sparse sampling, because it uses every window while still spending the alarm
budget once, and we could not evaluate it. ESPset records are independent measurements
with no acquisition clock, so any rolling rule there measures the order of a file.

CIRA has a clock. Three industrial pumps feeding boilers at a research centre, sampled
once a second over three operational days through a wireless mesh. This script
commissions the gate on each pump's earliest day and runs it forward over the later
days, in order, exactly as a deployed node would.

**What this can and cannot establish.** The data carry no labels. They are operational
records from a working plant, so we treat them as healthy and read every escalation as
a false alarm. That makes the measured rate an *upper bound*: if a pump was quietly
degrading, some of those escalations were correct and the true false-alarm rate is
lower. Nothing here bounds recall, and no claim about detection should be drawn from it.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message="Unknown solver options")


def window_channels(record, window_s: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce 1 Hz telemetry to per-window channel means.

    Gaps are averaged around rather than interpolated: the dropout is real and a node
    would face it. A window with no finite sample in a channel is dropped whole, since
    imputing it would invent the very quantity being tested.
    """
    n = len(record.timestamps)
    edges = range(0, n - window_s + 1, window_s)
    feats, stamps = [], []
    for s in edges:
        block = record.values[s : s + window_s]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(np.where(np.isfinite(block), block, np.nan), axis=0)
        if np.isfinite(m).all():
            feats.append(m)
            stamps.append(record.timestamps[s])
    if not feats:
        return np.empty((0, record.values.shape[1])), np.empty(0, dtype="datetime64[s]")
    return np.vstack(feats), np.array(stamps, dtype="datetime64[s]")


def persistence_alarms(escalations: np.ndarray, k: int, n: int) -> int:
    """Alarms raised by a 'k of the last n windows' rule, with re-arm.

    After firing, the trailing buffer is cleared so one sustained excursion produces one
    alarm rather than one per window. Counting every window of a single event as a
    separate alarm would flatter the sparse-sampling comparison.
    """
    fired, buf = 0, []
    for e in escalations:
        buf.append(bool(e))
        if len(buf) > n:
            buf.pop(0)
        if sum(buf) >= k:
            fired += 1
            buf = []
    return fired


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "cira")
    ap.add_argument("--window-s", type=int, default=60)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "results_cira.json")
    args = ap.parse_args()

    from pumpwatch.datasets.cira import (
        CHANNELS, CIRA_CITATION, CIRA_DOI, CIRA_LICENCE, load_cira,
    )
    from pumpwatch.node.gates import fit_composite_gate

    records = load_cira(args.root)
    by_pump: dict[str, list] = {}
    for r in sorted(records, key=lambda r: (r.pump_id, r.day)):
        by_pump.setdefault(r.pump_id, []).append(r)

    print(f"\n=== CIRA: {len(records)} pump-days, {len(by_pump)} pumps, "
          f"{args.window_s}s windows ===")
    print(f"    dropout {min(r.missing_fraction for r in records):.4f}"
          f"–{max(r.missing_fraction for r in records):.4f} of samples per pump-day")

    out = {"_meta": {
        "dataset": "cira", "real_data": True, "labelled": False,
        "doi": CIRA_DOI, "licence": CIRA_LICENCE, "citation": CIRA_CITATION,
        "window_s": args.window_s, "n_pumps": len(by_pump),
        "n_pump_days": len(records),
        "interpretation": (
            "Unlabelled operational data treated as healthy, so every escalation is "
            "counted as a false alarm. The measured rate is therefore an UPPER BOUND: "
            "a quietly degrading pump would make some of these escalations correct. "
            "Nothing here bounds recall."
        ),
    }}

    # Feature sets in increasing order of load-independence. The gate is commissioned
    # once on the earliest day and run forward, which is the deployment model, so any
    # channel that tracks plant demand rather than machine health will drift out of its
    # commissioned envelope and escalate continuously.
    idx = {c: i for i, c in enumerate(CHANNELS)}
    FEATURE_SETS = {
        "all_channels": [c for c in CHANNELS],
        "vibration_only": [c for c in CHANNELS if c.startswith("ACR")],
        "vibration_no_temperature": [
            c for c in CHANNELS if c.startswith("ACR") and not c.endswith("TV")
        ],
        "displacement_only": [
            c for c in CHANNELS if c.startswith("ACR") and c.endswith("PV")
        ],
        "dimensionless_ratio": ["ratio_pmp_mot_peak"],
    }

    def build(record, cols):
        X, _ = window_channels(record, args.window_s)
        if len(X) == 0:
            return X
        if cols == ["ratio_pmp_mot_peak"]:
            num, den = X[:, idx["ACR_Pmp.SV"]], np.maximum(X[:, idx["ACR_Mot.SV"]], 1e-9)
            return (num / den).reshape(-1, 1)
        return X[:, [idx[c] for c in cols]]

    print("\n=== How far does a once-commissioned gate drift? ===")
    print("  Commissioned on each pump's earliest day, run forward over the later ones.")
    print("  Every escalation counted as a false alarm (data presumed healthy).")
    print(f"\n  {'feature set':28}" + "".join(f"{p:>9}" for p in sorted(by_pump)) + "   mean")
    ablation = {}
    for label, cols in FEATURE_SETS.items():
        rates = {}
        for pump, days in sorted(by_pump.items()):
            if len(days) < 2:
                continue
            Xc = build(days[0], cols)
            need = int(np.ceil(10 * (Xc.shape[1] if len(Xc) else 1) * 1.5))
            if len(Xc) < need:
                continue
            gate = fit_composite_gate(Xc, feature_names=list(cols))
            esc = [
                bool(gate.update(x)["escalate"])
                for d in days[1:] for x in build(d, cols)
            ]
            if esc:
                rates[pump] = float(np.mean(esc))
        if rates:
            mean = float(np.mean(list(rates.values())))
            ablation[label] = {"per_pump": rates, "mean_escalation_rate": mean,
                               "n_features": len(cols)}
            print(f"  {label:28}" + "".join(f"{rates.get(p, float('nan')):>9.3f}"
                                            for p in sorted(by_pump)) + f"{mean:>8.3f}")
    out["feature_set_ablation"] = ablation

    # Persistence, on the least load-coupled feature set. If a k-of-n rule cannot rescue
    # the best case it cannot rescue the others.
    best = min(ablation, key=lambda k: ablation[k]["mean_escalation_rate"])
    print(f"\n=== Persistence on the best feature set ({best}) ===")
    print("  ESPset has no acquisition clock, so this rule could not be evaluated there.")
    cols = FEATURE_SETS[best]
    per_pump = {}
    for pump, days in sorted(by_pump.items()):
        if len(days) < 2:
            continue
        Xc = build(days[0], cols)
        if len(Xc) < int(np.ceil(10 * Xc.shape[1] * 1.5)):
            continue
        gate = fit_composite_gate(Xc, feature_names=list(cols))
        esc, hours = [], 0.0
        for d in days[1:]:
            Xe = build(d, cols)
            esc.extend(bool(gate.update(x)["escalate"]) for x in Xe)
            hours += len(Xe) * args.window_s / 3600.0
        esc = np.asarray(esc, dtype=bool)
        if esc.size == 0 or hours == 0:
            continue
        rules = {}
        for k, n in ((1, 1), (2, 3), (3, 5), (5, 10)):
            fired = persistence_alarms(esc, k, n)
            rules[f"{k}of{n}"] = {
                "alarms": int(fired),
                "alarms_per_month_at_3h_day": fired / hours * 3.0 * 30.0,
                "added_latency_min": (n - 1) * args.window_s / 60.0,
            }
        per_pump[pump] = {
            "commissioning_day": days[0].day,
            "evaluation_days": [d.day for d in days[1:]],
            "evaluation_hours": hours,
            "window_escalation_rate": float(esc.mean()),
            "persistence": rules,
        }
    out["per_pump"] = per_pump
    out["_meta"]["persistence_feature_set"] = best

    if per_pump:
        print(f"  {'rule':>8}{'mean alarms/month':>20}{'added latency':>16}{'vs 1/month':>14}")
        for rule in ("1of1", "2of3", "3of5", "5of10"):
            vals = [p["persistence"][rule]["alarms_per_month_at_3h_day"] for p in per_pump.values()]
            lat = next(iter(per_pump.values()))["persistence"][rule]["added_latency_min"]
            mean = float(np.mean(vals))
            print(f"  {rule:>8}{mean:>20.0f}{lat:>13.0f} min"
                  f"{('within' if mean <= 1 else f'{mean:.0f}x over'):>14}")
            out.setdefault("summary", {})[rule] = {"mean_alarms_per_month": mean}

    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    print(f"\n{CIRA_CITATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
