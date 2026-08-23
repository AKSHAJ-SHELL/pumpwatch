#!/usr/bin/env python3
"""Gate behaviour on real industrial telemetry, under a run-state-gated protocol.

This script previously reported that the gate escalates 100 % of windows on these
pumps and attributed it to plant demand moving the healthy baseline. **That was wrong,
and the error was ours.** The day used for commissioning is 89 %, 92 % and 85 % idle for
pumps A, B and C; the gate learned a stopped pump as normal and then saw a running one.
The corrected protocol commissions and evaluates on running windows only
(``pumpwatch.node.runstate``), and reports a much weaker and partly undecidable picture.

What this dataset can and cannot support, stated up front because the first version of
this script overstated it:

  * It has no labels. Operational records from a working plant are *presumed* healthy,
    so an escalation rate here is an upper bound on false alarms, never a measurement of
    them, and nothing here bounds recall.
  * Degradation and drift are indistinguishable without labels. If a pump's vibration
    triples over four months, that is either baseline drift or a developing bearing
    fault, and no amount of analysis of unlabelled data decides which.
  * Therefore: **unlabelled operational data can falsify a gate but cannot validate
    one.** A 100 % escalation rate proved our protocol was broken. A low rate would have
    proved nothing.

The value of running it is the protocol lessons, not the numbers: run state must be
detected before anything else, and commissioning must be counted in observed running
windows rather than assumed from a nominal duty.
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

    from pumpwatch.baseline_lifecycle import commissioning_progress
    from pumpwatch.node.runstate import RunState, RunStateDetector, config_from_healthy_load

    idx = {c: i for i, c in enumerate(CHANNELS)}
    LOAD = "Pres.PV"          # outlet pressure: the pump is doing work or it is not
    FEATURE_SETS = {
        "all_channels": list(CHANNELS),
        "vibration_no_temperature": [
            c for c in CHANNELS if c.startswith("ACR") and not c.endswith("TV")
        ],
    }

    def windows(record):
        X, _ = window_channels(record, args.window_s)
        return X

    # Run-state thresholds are derived per pump from its own load channel, pooled over
    # every day, because the units are plant-specific and a hardcoded bar figure would
    # not transfer. Pooling is legitimate here: it uses no labels, only the shape of the
    # load distribution, which is the same information a commissioning engineer has.
    print("\n=== Run state (threshold derived per pump from its own load channel) ===")
    print(f"  {'record':16}{'running':>9}{'of':>7}{'  idle share':>13}")
    per_day, run_cfg = {}, {}
    for pump, days in sorted(by_pump.items()):
        pooled = np.concatenate([windows(d)[:, idx[LOAD]] for d in days])
        try:
            run_cfg[pump] = config_from_healthy_load(pooled)
        except ValueError as exc:
            print(f"  [skip] {pump}: {exc}")
            continue
        for d in days:
            X = windows(d)
            det = RunStateDetector(run_cfg[pump])
            states = np.array([det.update(v) for v in X[:, idx[LOAD]]], dtype=object)
            mask = np.array([s == RunState.RUNNING for s in states], dtype=bool)
            per_day[(pump, d.day)] = (X, mask)
            print(f"  {pump}_{d.day:12}{int(mask.sum()):>9}{len(X):>7}{1 - mask.mean():>13.2f}")

    print("\n=== Commissioning: counted in observed RUNNING windows ===")
    out["per_pump"] = {}
    for label, cols in FEATURE_SETS.items():
        print(f"\n  [{label}] {len(cols)} features")
        for pump, days in sorted(by_pump.items()):
            blocks = [(d.day, *per_day[(pump, d.day)]) for d in days if (pump, d.day) in per_day]
            if not blocks:
                continue
            # First day that is actually commissioned, not the first day available.
            chosen = None
            for day, X, mask in blocks:
                prog = commissioning_progress(int(mask.sum()), len(cols))
                if prog.commissioned:
                    chosen = (day, X[mask][:, [idx[c] for c in cols]], prog)
                    break
            if chosen is None:
                worst = max(int(m.sum()) for _, _, m in blocks)
                req = commissioning_progress(worst, len(cols)).required_windows
                print(f"    {pump}: UNCOMMISSIONABLE — best day has {worst} running "
                      f"windows, needs {req}")
                out["per_pump"].setdefault(pump, {})[label] = {
                    "status": "uncommissionable",
                    "best_running_windows": worst, "required_windows": req,
                }
                continue

            day_c, Xc, prog = chosen
            gate = fit_composite_gate(Xc, feature_names=list(cols))
            # Evaluate only on days AFTER the commissioning day. Evaluating on earlier
            # ones inverts the deployment order — a node cannot be commissioned in June
            # and deployed in April — and would quietly reuse the idle April data the
            # original protocol tripped over.
            esc, hours = [], 0.0
            for day, X, mask in blocks:
                if day <= day_c:
                    continue
                Xr = X[mask][:, [idx[c] for c in cols]]
                esc.extend(bool(gate.update(x, run_state=RunState.RUNNING)["escalate"])
                           for x in Xr)
                hours += len(Xr) * args.window_s / 3600.0
            if not esc:
                continue
            esc = np.asarray(esc, dtype=bool)
            rules = {}
            for k, n in ((1, 1), (2, 3), (3, 5), (5, 10)):
                fired = persistence_alarms(esc, k, n)
                rules[f"{k}of{n}"] = {
                    "alarms": int(fired),
                    "alarms_per_month_at_3h_day": fired / hours * 3.0 * 30.0 if hours else None,
                    "added_latency_min": (n - 1) * args.window_s / 60.0,
                }
            out["per_pump"].setdefault(pump, {})[label] = {
                "status": "commissioned",
                "commissioning_day": day_c,
                "commissioning_windows": prog.observed_running_windows,
                "required_windows": prog.required_windows,
                "evaluation_running_windows": int(esc.size),
                "evaluation_hours": hours,
                "escalation_rate": float(esc.mean()),
                "persistence": rules,
            }
            print(f"    {pump}: commissioned on {day_c} ({prog.note}), "
                  f"escalation {esc.mean():.3f} over {esc.size} running windows")

    print("\n=== What this does and does not establish ===")
    print("  Unlabelled data. Every escalation counted as a false alarm, so these are")
    print("  UPPER BOUNDS. A pump whose vibration has tripled may be degrading, in which")
    print("  case its escalations are correct and the bound says nothing about the gate.")
    for pump, entries in sorted(out["per_pump"].items()):
        e = entries.get("vibration_no_temperature") or next(iter(entries.values()))
        if e["status"] != "commissioned":
            print(f"  {pump}: {e['status']} — reports no rate")
            continue
        peaks = []
        for d in by_pump[pump]:
            X = windows(d)
            m = per_day[(pump, d.day)][1]
            if m.any():
                peaks.append((d.day, float(np.median(X[m][:, idx['ACR_Mot.SV']]))))
        trend = " -> ".join(f"{v:.1f}" for _, v in peaks)
        ratio = peaks[-1][1] / peaks[0][1] if len(peaks) > 1 and peaks[0][1] else float("nan")
        verdict = ("UNDECIDABLE (median vibration x%.1f across the record — degradation "
                   "and drift are indistinguishable without labels)" % ratio
                   if ratio > 2.0 else "usable as an upper bound")
        print(f"  {pump}: escalation {e['escalation_rate']:.3f}; "
              f"median motor peak {trend}; {verdict}")
        out["per_pump"][pump]["median_motor_peak_by_day"] = peaks
        out["per_pump"][pump]["verdict"] = verdict

    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    print(f"\n{CIRA_CITATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
