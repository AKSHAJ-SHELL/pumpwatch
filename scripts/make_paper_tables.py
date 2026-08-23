#!/usr/bin/env python3
"""Emit the paper's results tables as markdown, read from results/*.json.

Every number in the write-up comes through here rather than being copied by hand.
That is the same discipline make_figures.py enforces for the plots, and for the same
reason: a hand-copied table silently goes stale the moment an experiment is re-run,
and a stale table in a paper is not recoverable after submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LADDER_LABELS = {
    0: "0 random-window (INVALID)",
    1: "1 record-wise",
    2: "2 component-wise",
    3: "3 cross-operating",
    4: "4 leave-one-machine-out",
}
MODEL_ORDER = ["majority", "logistic", "lightgbm", "tabpfn_abstain", "tabpfn_noabstain"]


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def ladder_table(results: dict, title: str) -> str:
    """One row per leakage level, one column per model: macro-F1 with its CI."""
    rows = {}
    for key, val in results.items():
        if not key.startswith("ladder__") or not isinstance(val, dict):
            continue
        lvl = val.get("split_level")
        rows.setdefault(lvl, {})[val.get("model")] = val

    models = [m for m in MODEL_ORDER if any(m in r for r in rows.values())]
    out = [f"**{title}** — macro-F1, 95% CI over held-out groups in brackets.", ""]
    out.append("| Leakage level | " + " | ".join(models) + " |")
    out.append("|---" * (len(models) + 1) + "|")
    for lvl in sorted(rows):
        cells = []
        for m in models:
            v = rows[lvl].get(m)
            if v is None:
                cells.append("—")
                continue
            ci = v.get("macro_f1_bootstrap_ci") or {}
            cell = _fmt(v.get("overall_macro_f1"))
            if ci.get("lo") is not None:
                cell += f" [{_fmt(ci['lo'], 2)}–{_fmt(ci['hi'], 2)}]"
            if (v.get("overall_coverage") or 1.0) < 0.999:
                cell += f" (cov {_fmt(v['overall_coverage'], 2)})"
            cells.append(cell)
        label = LADDER_LABELS.get(lvl, str(lvl))
        verdict = next(iter(rows[lvl].values())).get("split_verdict")
        if verdict:
            label += f" — *{verdict}*"
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def leakage_inflation(results: dict) -> str:
    """The headline of C5: what the invalid split buys you, per model."""
    by = {}
    for key, val in results.items():
        if key.startswith("ladder__") and isinstance(val, dict):
            by.setdefault(val.get("model"), {})[val.get("split_level")] = val.get(
                "overall_macro_f1"
            )
    lines = ["| Model | random-window | strictest valid | inflation |", "|---|---|---|---|"]
    for m in MODEL_ORDER:
        d = by.get(m) or {}
        invalid = d.get(0)
        valid_levels = [lv for lv in d if lv and lv > 0 and d[lv] is not None]
        if invalid is None or not valid_levels:
            continue
        strictest = d[max(valid_levels)]
        ratio = invalid / strictest if strictest else None
        lines.append(
            f"| {m} | {_fmt(invalid)} | {_fmt(strictest)} | "
            f"{'—' if ratio is None else f'{ratio:.1f}x'} |"
        )
    return "\n".join(lines)


def normalisation_table(results: dict) -> str:
    """Both normalisation strategies side by side, per model.

    These must never be quoted interchangeably. The choice is transductive vs
    inductive - whether the held-out machine's own statistics were used to normalise
    it - and on real data it is worth more than the choice of model. A table that
    shows one strategy in one section and the other in the next, unlabelled, is the
    silent incomparability this project exists to avoid.
    """
    by = {}
    for key, val in results.items():
        if "__" in key and not key.startswith("ladder__") and isinstance(val, dict):
            if "overall_macro_f1" not in val:
                continue
            model, _, strat = key.partition("__")
            by.setdefault(model, {})[strat] = val

    if not by:
        return "_No per-strategy results in this file._"
    strats = ["unsupervised_per_machine", "train_pooled"]
    lines = [
        "| Model | " + " | ".join(strats) + " | delta |",
        "|---" * (len(strats) + 2) + "|",
    ]
    for m in MODEL_ORDER:
        d = by.get(m)
        if not d:
            continue
        cells = []
        for s in strats:
            v = d.get(s)
            cell = "—" if v is None else _fmt(v.get("overall_macro_f1"))
            if v is not None and (v.get("overall_coverage") or 1.0) < 0.999:
                cell += f" (cov {_fmt(v['overall_coverage'], 2)})"
            cells.append(cell)
        a = (d.get(strats[0]) or {}).get("overall_macro_f1")
        b = (d.get(strats[1]) or {}).get("overall_macro_f1")
        delta = "—" if a is None or b is None else f"{b - a:+.3f}"
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {delta} |")
    lines += [
        "",
        "> `unsupervised_per_machine` normalises each machine using its own statistics, "
        "including the held-out one - transductive, and legitimate when a node "
        "self-commissions on the target pump. `train_pooled` uses training-machine "
        "statistics only - inductive, and the stricter reading. **Never quote one for "
        "the other.** The leakage-ladder tables above use `unsupervised_per_machine` "
        "throughout.",
    ]
    return "\n".join(lines)


def gate_table(results: dict) -> str:
    summary = results.get("gate_summary") or {}
    stage1 = results.get("gate_stage1") or {}
    if not summary and not stage1:
        return "_No gate stage in this results file._"
    lines = ["| Pump | healthy escalation | faulty escalation | field rate | commissioned |",
             "|---|---|---|---|---|"]
    for pump, v in sorted(stage1.items()):
        lines.append(
            f"| {pump} | {_fmt(v.get('escalation_rate_healthy'), 2)} | "
            f"{_fmt(v.get('escalation_rate_faulty'), 2)} | "
            f"{_fmt(v.get('escalation_rate_field'), 3)} | "
            f"{'yes' if v.get('commissioning_adequate', True) else 'NO'} |"
        )
    if summary:
        lines += [
            "",
            f"Field-weighted escalation **{_fmt(summary.get('mean_field_escalation_rate'))}**, "
            f"gate recall ceiling **{_fmt(summary.get('gate_recall_ceiling'), 2)}**, "
            f"{_fmt(summary.get('uplinks_per_day_at_field_rate'), 1)} uplinks/day, "
            f"{_fmt(summary.get('battery_years_at_field_rate'), 2)} yr battery, "
            f"TX {_fmt(100 * (summary.get('tx_fraction') or 0), 1)}% of the budget. "
            f"Adequately commissioned on "
            f"{summary.get('n_machines_adequately_commissioned')}/"
            f"{summary.get('n_machines')} machines.",
            "",
            "> Gateway accuracy is an upper bound conditioned on escalation: end-to-end "
            "fault recall cannot exceed the gate recall ceiling. Battery life is driven "
            "by the *field* rate, which healthy false-escalation dominates — the "
            "test-set rate reflects how many faulty examples were collected, not field "
            "prevalence.",
        ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "paper_tables.md")
    args = ap.parse_args()

    sources = [
        ("results_espset_both.json", "ESPset — 11 in-service submersible pumps"),
        ("results_twente_real.json", "Twente/4TU — 2 motors, 4 operating speeds"),
        ("results_full.json", "Synthetic stand-in — wiring check, NOT a result"),
    ]
    parts = ["# Paper tables (generated — do not hand-edit)", ""]
    for fname, title in sources:
        path = ROOT / "results" / fname
        if not path.exists():
            parts += [f"## {title}", "", f"_Missing: `results/{fname}`._", ""]
            continue
        results = json.loads(path.read_text())
        meta = results.get("_meta", {})
        parts += [f"## {title}", ""]
        if meta.get("interpretation_caveat"):
            parts += [f"> ⚠️ {meta['interpretation_caveat']}", ""]
        n = [f"{k}={meta[k]}" for k in ("n_samples", "n_records", "n_features", "n_machines")
             if k in meta]
        if n:
            parts += ["`" + ", ".join(n) + "`", ""]
        parts += [ladder_table(results, "Leakage ladder"), "", "**Leakage inflation**", "",
                  leakage_inflation(results), "",
                  "**Normalisation strategy** (cross-machine)", "",
                  normalisation_table(results), "", "**Stage-1 gate**", "",
                  gate_table(results), ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n")
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
