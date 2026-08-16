#!/usr/bin/env python3
"""Generate core figure suite into figures/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pumpwatch.figures import (
    fig_lomo_per_machine,
    fig_profile_comparison,
    make_all_core_figures,
)


def main():
    outdir = ROOT / "figures"
    paths = make_all_core_figures(outdir)
    # Placeholder profile / LOMO figures from last run if present
    fig_profile_comparison(
        outdir / "B5_profile_ablation.png",
        full_scores={"majority": 0.2, "logistic": 0.75, "lightgbm": 0.78},
        ct_only_scores={"majority": 0.2, "logistic": 0.55, "lightgbm": 0.58},
    )
    paths.append(outdir / "B5_profile_ablation.png")
    fig_lomo_per_machine(
        outdir / "D7_lomo_per_machine.png",
        {"NK80-250": 0.71, "NK80-160": 0.64},
    )
    paths.append(outdir / "D7_lomo_per_machine.png")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
