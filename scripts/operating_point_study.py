#!/usr/bin/env python3
"""What would it take to make this system deployable?

End-to-end recall at the design's stated operating point is 0.17: the gate discards
some faults, and the classifier at a false-alarm rate of 0.00093 recovers only 0.20 of
what reaches it. That is not a usable detector, and this script asks which of the
levers available actually moves it.

The false-alarm rate is not a physical constant. It comes from a duty-cycle choice -
12 feature windows per runtime hour, three runtime hours a day - which puts ~1080
decisions in a pump-month, so one tolerated alarm a month divides down to 0.00093 per
decision. Deciding less often, or requiring a fault to persist across several windows
before alarming, both buy false-alarm headroom in exchange for detection latency. For
the slow faults this tier exists to catch - bearing wear, misalignment, impeller damage
developing over weeks - latency is close to free.

Sweeps three things and reports what each is worth:
  1. decision rate: how much recall a looser per-decision budget buys
  2. persistence: requiring k of the last n windows to agree before alarming
  3. task difficulty: multi-class diagnosis versus binary fault detection
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
warnings.filterwarnings("ignore", message=".*OOD abstention disabled.*")


def persistence_filter(scores, groups, k, n):
    """Alarm only if k of the last n windows from the same machine exceed threshold.

    Implemented on scores rather than decisions: the k-th largest of the trailing
    window is the score at which a k-of-n rule would fire, so thresholding it is
    exactly equivalent and keeps the FAR sweep meaningful.
    """
    scores = np.asarray(scores, dtype=float)
    out = np.full_like(scores, -np.inf)
    for g in set(groups):
        idx = np.flatnonzero(np.asarray(groups) == g)
        s = scores[idx]
        for j in range(len(s)):
            lo = max(0, j - n + 1)
            window = s[lo : j + 1]
            if len(window) >= k:
                out[idx[j]] = np.sort(window)[-k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "operating_point_study.json")
    args = ap.parse_args()

    from pumpwatch.datasets.espset import espset_order_features, load_espset
    from pumpwatch.evaluate import fault_score_from_proba, recall_at_fixed_far
    from pumpwatch.gateway.baselines import fit_predict
    from pumpwatch.models import build_model_zoo
    from pumpwatch.splits import normalize_features, split_lomo

    data = load_espset(ROOT / "data" / "espset")
    Xo, _ = espset_order_features(data)
    X = np.hstack([Xo, data.published_features])
    y = np.asarray(data.labels)
    machines = np.asarray(data.machine_ids)

    zoo = build_model_zoo(include_tabpfn=args.model.startswith("tabpfn"), verbose=False)
    factory = zoo[args.model]
    split = split_lomo(machines.tolist())

    # Pooled out-of-fold fault scores under the better normalisation strategy.
    scores = np.zeros(len(y)); order = []
    for fold in split.folds:
        tr, te = fold.train_idx, fold.test_idx
        Xn = normalize_features(X, machines.tolist(), tr, strategy="train_pooled")
        res = fit_predict(factory(0), Xn[tr], y[tr], Xn[te], args.model)
        scores[te] = fault_score_from_proba(res.y_proba, res.classes, healthy_label="healthy")
        order.extend(te.tolist())
    is_fault = (y != "healthy").astype(int)

    out = {}
    print("\n=== 1. DECISION RATE — how much does deciding less often buy? ===")
    print("  A month has 1080 decisions at the design duty. Fewer decisions means a")
    print("  looser per-decision budget for the same one-alarm-a-month promise.")
    print(f"\n  {'decisions/month':>16}{'cadence':>22}{'FAR budget':>13}{'recall':>9}")
    rate_rows = {}
    for n_dec, cadence in ((1080, "every 5 min (design)"), (360, "every 15 min"),
                           (90, "hourly"), (30, "daily"), (12, "every 2.5 days")):
        far = 1.0 / n_dec
        r = recall_at_fixed_far(is_fault, scores, far)
        rate_rows[n_dec] = {"far": far, "recall": r["recall"], "cadence": cadence}
        print(f"  {n_dec:>16}{cadence:>22}{far:>13.5f}{r['recall']:>9.3f}")
    out["decision_rate"] = rate_rows

    # Persistence (k of the last n windows) is NOT evaluated here. ESPset records are
    # independent measurements with no time axis, so a "trailing window" is an
    # arbitrary ordering and any rolling rule measures the sort order, not the physics.
    # An earlier version of this script reported such numbers; they were meaningless.
    # Evaluating persistence needs a dataset with real acquisition timestamps.
    pers = {"note": "not evaluable on ESPset - no time axis; requires timestamped data"}
    out["persistence"] = pers
    print("\n=== 2. PERSISTENCE — NOT EVALUABLE HERE ===")
    print("  ESPset records are independent measurements with no time axis, so a")
    print("  k-of-n rule over a 'trailing window' would measure sort order, not physics.")
    print("  This lever is plausible but untested; it needs timestamped acquisitions.")

    print("\n=== 3. TASK — diagnosis vs detection ===")
    print("  Multi-class diagnosis is dragged down by misalignment (n=70, F1 0.55).")
    print("  Binary 'is this pump faulty' is the question a farmer actually asks.")
    for far, lab in ((1/1080, "design budget"), (1/30, "daily decisions")):
        r = recall_at_fixed_far(is_fault, scores, far)
        print(f"  binary detection @ {lab:16} FAR={far:.5f}  recall={r['recall']:.3f}")
        out.setdefault("binary_detection", {})[lab] = r["recall"]

    gate_ceiling = json.load(open(ROOT / "results" / "results_espset_both.json"))[
        "gate_summary"]["gate_recall_ceiling"]
    print(f"\n=== END-TO-END, including the gate ceiling of {gate_ceiling:.2f} ===")
    print(f"  as designed          : {gate_ceiling * rate_rows[1080]['recall']:.3f}")
    print(f"  daily decisions      : {gate_ceiling * rate_rows[30]['recall']:.3f}")
    print(f"  every 2.5 days       : {gate_ceiling * rate_rows[12]['recall']:.3f}")

    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
