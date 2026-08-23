#!/usr/bin/env python3
"""Does the normalisation effect replicate on the second dataset?

On eleven in-service pumps, standardising each machine by its own statistics costs
0.08 to 0.25 macro-F1 against pooling the training machines - more than the gap between
any two models compared here. Before that can headline a paper it needs a replication,
and the honest position is that a full one is impossible: Twente cannot support
leave-one-machine-out at all, because its two motors share no fault class.

What Twente *can* do is the other rungs. If the normalisation choice matters at
record-wise, component-wise and cross-operating splits on a second, independently
collected dataset with a different sensor suite, the effect is not an ESPset artefact
even though the cross-machine claim still rests on one dataset. If it does not, the
headline must be tempered.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message="Unknown solver options")


def main() -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "twe_exp", ROOT / "scripts" / "run_twente_experiment.py"
    )
    twe = importlib.util.module_from_spec(spec)
    sys.argv = ["run_twente_experiment.py"]
    try:
        spec.loader.exec_module(twe)
    except SystemExit:
        pass

    from pumpwatch.datasets.twente_raw import load_twente_raw
    from pumpwatch.experiment import build_ladder, run_split
    from pumpwatch.models import build_model_zoo

    records = load_twente_raw(ROOT / "data" / "raw" / "twente_sel")
    X, y, machines, names, groups, _sev = twe.build_table(records, "full")
    ladder = build_ladder(machines, groups, len(y))

    zoo = build_model_zoo(include_tabpfn=False, verbose=False)
    models = {k: v for k, v in zoo.items() if k in ("logistic", "lightgbm")}

    print(f"\n  Twente: {len(y)} records, {len(set(machines))} machines, "
          f"{X.shape[1]} features")
    print(f"\n  {'rung':22}{'model':11}{'per-machine':>12}{'pooled':>9}{'gap':>9}")
    out = {}
    for rung, split in sorted(ladder.items()):
        if getattr(split, "verdict", "") == "INVALID":
            continue                       # the invalid rung tells us nothing here
        for name, factory in models.items():
            try:
                a = run_split(X, y, machines, factory, name, split,
                              norm_strategy="unsupervised_per_machine")["overall_macro_f1"]
                b = run_split(X, y, machines, factory, name, split,
                              norm_strategy="train_pooled")["overall_macro_f1"]
            except Exception as exc:
                print(f"  {rung:22}{name:11} skipped: {type(exc).__name__}")
                continue
            out.setdefault(rung, {})[name] = {"per_machine": a, "train_pooled": b, "gap": b - a}
            print(f"  {rung:22}{name:11}{a:>12.3f}{b:>9.3f}{b - a:>+9.3f}")

    gaps = [v["gap"] for r in out.values() for v in r.values()]
    if gaps:
        pos = sum(1 for g in gaps if g > 0)
        print(f"\n  pooled beats per-machine on {pos}/{len(gaps)} rung-model pairs; "
              f"mean gap {np.mean(gaps):+.3f}")
        print("  Direction agreeing with ESPset supports the effect being general.")
        print("  Note this is NOT a cross-machine replication - Twente cannot do LOMO.")

    p = ROOT / "results" / "normalisation_replication.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
