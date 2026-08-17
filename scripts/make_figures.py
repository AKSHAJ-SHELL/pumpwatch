#!/usr/bin/env python3
"""Generate the figure suite into figures/.

Every result figure is built from a results JSON written by run_experiment.py.
Nothing here hardcodes a score. An earlier version did — it shipped B5 and D7
showing macro-F1 around 0.75 while the actual committed run was 0.028 for every
model, because the figures never read the results file. If the results are absent
this script fails rather than inventing numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pumpwatch.figures import (
    fig_accuracy_vs_latency,
    fig_context_sweep,
    fig_recall_at_alarm_budget,
    fig_calibration,
    fig_energy_breakdown,
    fig_escalation_vs_battery,
    fig_leakage_ladder,
    fig_lomo_per_machine,
    fig_normalization_gap,
    fig_profile_comparison,
    fig_tabpfn_latency,
    make_all_core_figures,
)


class ResultsMissingError(FileNotFoundError):
    def __init__(self, path: Path):
        super().__init__(
            f"No results at {path}.\n"
            "Run `make experiment` first. Result figures are built from measured "
            "numbers only — this script will not synthesise placeholder scores."
        )


def load_results(path: Path) -> dict:
    if not path.exists():
        raise ResultsMissingError(path)
    return json.loads(path.read_text())


def _macro_f1_by_model(results: dict, suffix: str) -> dict[str, float]:
    """Pull {model: overall_macro_f1} for every key ending in `suffix`."""
    out = {}
    for key, val in results.items():
        if not isinstance(val, dict) or "overall_macro_f1" not in val:
            continue
        if key.endswith(suffix):
            out[key[: -len(suffix)].rstrip("_")] = val["overall_macro_f1"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "results_full.json")
    parser.add_argument("--outdir", type=Path, default=ROOT / "figures")
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help="Emit only the physics/energy/trip figures that need no results file.",
    )
    args = parser.parse_args()

    paths = make_all_core_figures(args.outdir)

    if args.physics_only:
        for p in paths:
            print(p)
        return

    results = load_results(args.results)
    strategy = "unsupervised_per_machine"

    # C5 / E3 — the gate's measured escalation rate drives the energy figures.
    gate = results.get("gate_summary") or {}
    measured_rate = gate.get("mean_field_escalation_rate")
    paths.append(
        fig_escalation_vs_battery(
            args.outdir / "C5_escalation_vs_battery.png", measured_rate=measured_rate
        )
    )
    if measured_rate is not None:
        paths.append(
            fig_energy_breakdown(
                args.outdir / "E3_energy_breakdown.png", escalation_rate=measured_rate
            )
        )

    # D4 — context-size sweep: how many labelled windows commissioning needs.
    sweep = results.get("tabpfn_context_sweep")
    if sweep:
        paths.append(fig_context_sweep(args.outdir / "D4_context_sweep.png", sweep))

    # Recall at the farmer-facing alarm budget.
    budget = results.get("recall_at_alarm_budget")
    if budget and any(budget.values()):
        paths.append(
            fig_recall_at_alarm_budget(args.outdir / "D12_recall_at_alarm_budget.png", budget)
        )

    # E1 — measured TabPFN latency: KV cache and ensemble size.
    bench = results.get("tabpfn_benchmark")
    if bench:
        paths.append(fig_tabpfn_latency(args.outdir / "E1_tabpfn_latency.png", bench))

    # D2 — accuracy against compute. Contribution C4: does the expensive model earn it?
    pts = []
    for key, val in results.items():
        if not isinstance(val, dict) or not key.endswith(f"__{strategy}"):
            continue
        if key.startswith("ct_only") or "overall_macro_f1" not in val:
            continue
        pts.append({
            "model": val["model"],
            "macro_f1": val["overall_macro_f1"],
            "latency_s": val.get("mean_latency_predict_s", 0.0),
        })
    if len(pts) > 1:
        paths.append(fig_accuracy_vs_latency(args.outdir / "D2_accuracy_vs_latency.png", pts))

    # D1 — macro-F1 vs split protocol. The leakage argument, measured.
    ladder: dict[str, dict[str, float]] = {}
    for key, val in results.items():
        if not key.startswith("ladder__") or not isinstance(val, dict):
            continue
        _, rung, model = key.split("__", 2)
        ladder.setdefault(rung, {})[model] = val["overall_macro_f1"]
    if ladder:
        paths.append(fig_leakage_ladder(args.outdir / "D1_leakage_ladder.png", ladder))

    # B5 — sensor profile ablation, measured.
    full_scores = _macro_f1_by_model(results, f"__{strategy}")
    full_scores = {k: v for k, v in full_scores.items() if "ct_only" not in k}
    ct_scores = {
        k.replace("ct_only__", ""): v
        for k, v in _macro_f1_by_model(results, f"__{strategy}").items()
        if k.startswith("ct_only__")
    }
    if full_scores and ct_scores:
        paths.append(
            fig_profile_comparison(
                args.outdir / "B5_profile_ablation.png", full_scores, ct_scores
            )
        )

    # D7 — LOMO per machine, measured, for the best non-majority model.
    candidates = {
        k: v for k, v in results.items()
        if isinstance(v, dict) and k.endswith(f"__{strategy}")
        and "per_machine_macro_f1" in v and not k.startswith("majority")
        and not k.startswith("ct_only")
    }
    if candidates:
        best = max(candidates.values(), key=lambda v: v["overall_macro_f1"])
        paths.append(
            fig_lomo_per_machine(
                args.outdir / "D7_lomo_per_machine.png",
                best["per_machine_macro_f1"],
                model_name=best["model"],
                strategy=best["norm_strategy"],
            )
        )

    # Normalisation-strategy gap: how much of the LOMO score needs the target pump's
    # own distribution. Transductive vs inductive, side by side.
    gap = {
        s: _macro_f1_by_model(results, f"__{s}")
        for s in ("unsupervised_per_machine", "train_pooled")
    }
    gap = {s: {k: v for k, v in d.items() if "ct_only" not in k} for s, d in gap.items()}
    if all(gap.values()):
        paths.append(fig_normalization_gap(args.outdir / "D11_normalization_gap.png", gap))

    # Calibration — TabPFN's central claim is a calibrated posterior. Only plotted
    # when a model actually produced probabilities.
    for key, val in results.items():
        if not isinstance(val, dict) or "per_machine" not in val:
            continue
        if not key.endswith(f"__{strategy}") or key.startswith(("majority", "ct_only")):
            continue
        eces = [m["ece"] for m in val["per_machine"].values() if m.get("ece") is not None]
        if eces:
            paths.append(
                fig_calibration(
                    args.outdir / f"D6_calibration_{val['model']}.png",
                    val["per_machine"],
                    label=val["model"],
                )
            )

    seen = set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            print(p)


if __name__ == "__main__":
    main()
