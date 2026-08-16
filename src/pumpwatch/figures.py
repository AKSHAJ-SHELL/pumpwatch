"""Figure suite — revised after red-team.

New headline figures:
  - false-trip vs detection delay (dry-run trip + valve confusers)
  - full vs ct_only accuracy
  - event-triggered battery life vs runtime hours (fixed schedule shown failing)

Cut: stage-1 hierarchy, multi-label, CD diagram at n=3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pumpwatch.node.energy import event_triggered_energy, fixed_schedule_energy
from pumpwatch.node.trip import (
    TripConfig,
    evaluate_trip_path,
    select_operating_point,
    sweep_trip_operating_points,
)
from pumpwatch.physics import cavitation_broadband_intensity


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_cavitation_nonmonotonic(out: Path) -> Path:
    xs = np.linspace(0, 1, 200)
    ys = np.array([cavitation_broadband_intensity(s) for s in xs])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, lw=2, color="C0")
    ax.axvline(0.4, ls="--", color="gray", label="developed peak")
    ax.set_xlabel("Cavitation severity")
    ax.set_ylabel("Relative broadband intensity")
    ax.set_title("Cavitation non-monotonicity (peak-and-fall)")
    ax.legend()
    return _save(fig, out)


def fig_trip_false_alarm(out: Path, n_trials: int = 40, seed: int = 0) -> Path:
    """Trip path against its confusers, at the selected operating point.

    Panel A contrasts the shipped AND rule with the OR rule it replaced — under OR
    the absolute floor was decorative and the trip fired on every throttled valve.
    Panel B is the detection-delay distribution against the seal-survival budget.
    """
    cfg = TripConfig()
    result = evaluate_trip_path(n_trials=n_trials, seed=seed, trip_config=cfg, duration_s=6.0)
    naive = evaluate_trip_path(
        n_trials=n_trials,
        seed=seed,
        trip_config=TripConfig(cusum_h=3.5, persistence_n=2, require_floor=False),
        duration_s=6.0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["dry-run\ndetect", "closed-valve\nfalse trip", "healthy\nfalse trip"]
    shipped = [
        result.dry_run_detection_rate,
        result.closed_valve_false_trip_rate,
        result.healthy_false_trip_rate,
    ]
    naive_vals = [
        naive.dry_run_detection_rate,
        naive.closed_valve_false_trip_rate,
        naive.healthy_false_trip_rate,
    ]
    x = np.arange(len(labels))
    w = 0.38
    axes[0].bar(x - w / 2, naive_vals, w, color="C3", label="CUSUM OR floor (rejected)")
    axes[0].bar(x + w / 2, shipped, w, color="C2", label="CUSUM AND floor (shipped)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Trip path vs confusers")
    axes[0].legend(fontsize=8)
    for xi, v in zip(x - w / 2, naive_vals):
        axes[0].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, shipped):
        axes[0].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    if result.delays_s:
        # Log axis: the delays are ~1 s and the seal budget is 60 s, so a linear
        # axis renders the distribution as a single invisible spike. The margin is
        # the point of the figure, so both have to be legible.
        bins = np.logspace(
            np.log10(max(min(result.delays_s), 1e-2)),
            np.log10(70.0),
            30,
        )
        axes[1].hist(result.delays_s, bins=bins, color="C0", edgecolor="white")
        axes[1].axvline(
            result.dry_run_median_delay_s,
            color="k",
            ls="--",
            label=f"median={result.dry_run_median_delay_s:.2f}s",
        )
    axes[1].axvspan(60.0, 70.0, color="C3", alpha=0.25)
    axes[1].axvline(60.0, color="C3", ls="-", label="seal survival limit (60 s)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Detection delay (s, log scale)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Dry-run detection delay vs seal budget")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def fig_trip_operating_points(out: Path, n_trials: int = 25, seed: int = 0) -> Path:
    """Detection vs false-trip across the parameter sweep, with the choice marked.

    The operating point is a safety decision with asymmetric costs, so it is picked
    off this curve rather than hardcoded in a config file.
    """
    points = sweep_trip_operating_points(n_trials=n_trials, seed=seed)
    chosen = select_operating_point(points)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    worst_false = [
        max(p["closed_valve_false_trip_rate"], p["healthy_false_trip_rate"])
        for p in points
    ]
    detect = [p["detection_rate"] for p in points]
    floors = [p["absolute_floor_fraction"] for p in points]

    sc = axes[0].scatter(worst_false, detect, c=floors, cmap="viridis", s=45, alpha=0.85)
    fig.colorbar(sc, ax=axes[0], label="absolute floor (fraction of rated)")
    if chosen:
        axes[0].scatter(
            [max(chosen["closed_valve_false_trip_rate"], chosen["healthy_false_trip_rate"])],
            [chosen["detection_rate"]],
            marker="*", s=420, facecolor="none", edgecolor="C3", linewidth=2,
            label="selected",
        )
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Worst false-trip rate (closed-valve or healthy)")
    axes[0].set_ylabel("Dry-run detection rate")
    axes[0].set_title("Trip operating points")
    axes[0].set_ylim(-0.05, 1.05)

    # Why the floor is the parameter that matters: it is the only one that knows
    # the difference between 45% of rated and 70% of rated.
    by_floor: dict[float, list[float]] = {}
    for p in points:
        by_floor.setdefault(p["absolute_floor_fraction"], []).append(
            p["closed_valve_false_trip_rate"]
        )
    xs = sorted(by_floor)
    axes[1].plot(xs, [np.mean(by_floor[f]) for f in xs], "o-", color="C3")
    axes[1].axvspan(0.45, 0.70, color="gray", alpha=0.18)
    axes[1].text(0.575, 0.5, "dry-run\n↔\nclosed-valve", ha="center", fontsize=8)
    axes[1].set_xlabel("Absolute floor (fraction of rated)")
    axes[1].set_ylabel("Closed-valve false-trip rate")
    axes[1].set_title("The floor is what separates the confuser")
    axes[1].set_ylim(-0.05, 1.05)
    fig.tight_layout()
    return _save(fig, out)


def fig_energy_battery_life(out: Path) -> Path:
    runtimes = [0.5, 1, 2, 3, 4, 6, 8]
    ev = [event_triggered_energy(h) for h in runtimes]
    # Fixed schedule comparisons
    fixed_15 = fixed_schedule_energy(900.0)
    fixed_1 = fixed_schedule_energy(60.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    years = [e.battery_years for e in ev]
    mAh = [e.mAh_per_day for e in ev]
    axes[0].plot(runtimes, years, "o-", label="event-triggered")
    axes[0].axhline(fixed_15.battery_years, ls="--", color="C1", label=f"fixed 15 min ({fixed_15.battery_years:.1f} yr)")
    axes[0].axhline(fixed_1.battery_years, ls=":", color="C3", label=f"fixed 1 min ({fixed_1.battery_years:.2f} yr)")
    axes[0].axhline(0.5, ls="-.", color="gray", label="6-month floor")
    axes[0].set_xlabel("Pump runtime (h/day)")
    axes[0].set_ylabel("Battery life (years, 2400 mAh usable)")
    axes[0].set_title("Battery life vs duty — event-triggered primary")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, max(years) * 1.1)

    axes[1].plot(runtimes, mAh, "o-", color="C0")
    axes[1].axhline(fixed_15.mAh_per_day, ls="--", color="C1", label="fixed 15 min")
    axes[1].set_xlabel("Pump runtime (h/day)")
    axes[1].set_ylabel("mAh / day")
    axes[1].set_title("Daily energy draw")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def fig_energy_breakdown(
    out: Path,
    runtime_h: float = 3.0,
    escalation_rate: Optional[float] = None,
) -> Path:
    """Per-phase energy for the event-triggered node.

    Reads the breakdown straight off the energy model. It used to re-derive the
    entire calculation by hand with its own hardcoded constants — two copies of the
    same arithmetic, free to drift apart.
    """
    result = event_triggered_energy(runtime_h, escalation_rate=escalation_rate)
    parts = result.breakdown_mAh
    total = sum(parts.values())

    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = list(parts.keys())
    vals = [parts[k] for k in labels]
    ax.barh(labels, vals, color=["C0", "C1", "C3", "C4", "C2"])
    ax.set_xlabel("mAh / day")
    subtitle = f"{runtime_h} h/day runtime"
    if escalation_rate is not None:
        subtitle += f", gate escalation {escalation_rate:.1%}"
    ax.set_title(f"Event-triggered energy breakdown ({subtitle})")
    ax.set_xlim(0, max(vals) * 1.35)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.2f} ({100 * v / total:.0f}%)", va="center", fontsize=8)
    return _save(fig, out)


def fig_escalation_vs_battery(
    out: Path,
    runtime_h: float = 3.0,
    measured_rate: Optional[float] = None,
) -> Path:
    """Gate escalation rate → transmissions/day → battery years.

    This is the quantitative content of the two-tier architecture claim: the gate
    exists to keep the radio quiet, and the radio dominates the energy budget. Two
    panels rather than a dual-y axis, which would invite reading a crossing point
    that means nothing.
    """
    rates = np.linspace(0.0, 1.0, 41)
    results = [event_triggered_energy(runtime_h, escalation_rate=float(r)) for r in rates]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rates * 100, [r.transmissions_per_day for r in results], "-", color="C0")
    axes[0].set_xlabel("Gate escalation rate (%)")
    axes[0].set_ylabel("Uplinks / day")
    axes[0].set_title("Escalation rate sets the radio duty")

    axes[1].plot(rates * 100, [r.battery_years for r in results], "-", color="C0")
    axes[1].axhline(1.0, ls=":", color="gray", label="1 year")
    axes[1].set_xlabel("Gate escalation rate (%)")
    axes[1].set_ylabel("Battery life (years, 2400 mAh usable)")
    axes[1].set_title("…and therefore battery life")

    if measured_rate is not None:
        for ax in axes:
            ax.axvline(
                measured_rate * 100, color="C3", ls="--",
                label=f"measured gate: {measured_rate:.1%}",
            )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def fig_profile_comparison(
    out: Path,
    full_scores: dict[str, float],
    ct_only_scores: dict[str, float],
) -> Path:
    """Headline: full (vib+CT) vs ct_only accuracy/F1."""
    models = sorted(set(full_scores) | set(ct_only_scores))
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [full_scores.get(m, 0) for m in models], w, label="full (vib+CT)")
    ax.bar(x + w / 2, [ct_only_scores.get(m, 0) for m in models], w, label="ct_only (submersible)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Sensor profile ablation — submersible reality check")
    ax.legend()
    ax.set_ylim(0, 1.05)
    return _save(fig, out)


def fig_lomo_per_machine(
    out: Path,
    per_machine: dict[str, float],
    model_name: str = "model",
    strategy: str = "",
) -> Path:
    """The thesis test. Each bar is one machine — with 2-3 machines these ARE the
    data points, so they are plotted individually rather than hidden behind a mean.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(per_machine.keys())
    vals = [per_machine[n] for n in names]
    ax.bar(names, vals, color="C0")
    ax.set_ylabel("Macro-F1")
    subtitle = f"{model_name}" + (f", norm={strategy}" if strategy else "")
    ax.set_title(f"LOMO per-machine ({subtitle})\nn machines = {len(names)} = the thesis unit")
    ax.set_ylim(0, 1.05)
    ax.axhline(float(np.mean(vals)), ls="--", color="k", label=f"mean={np.mean(vals):.2f}")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.legend(fontsize=8)
    return _save(fig, out)


def fig_leakage_ladder(out: Path, ladder: dict[str, dict[str, float]]) -> Path:
    """Macro-F1 against split protocol — the figure that makes the leakage argument.

    `ladder` maps rung name → {model: macro_f1}. The point is the drop from the
    random-window split (which is a memorisation test, not an evaluation) to
    leave-one-machine-out. A paper reporting only the left-hand bar is reporting
    how well the model memorised the recording it was trained on.
    """
    rungs = sorted(ladder)
    models = sorted({m for scores in ladder.values() for m in scores})
    x = np.arange(len(rungs))
    w = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, model in enumerate(models):
        ax.bar(
            x + (i - (len(models) - 1) / 2) * w,
            [ladder[r].get(model, 0.0) for r in rungs],
            w,
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([r.split("_", 1)[1].replace("_", " ") for r in rungs], rotation=12)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        "Leakage ladder: the same models, the same data, five split protocols",
        pad=22,
    )
    # Legend below the axes so it cannot sit on top of the verdict row.
    ax.legend(
        fontsize=8,
        ncol=len(models),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
    )

    verdicts = {
        "0_random_window": "INVALID",
        "1_record_wise": "weak",
        "2_component_wise": "good",
        "3_cross_operating": "essential",
        "4_lomo": "thesis test",
    }
    for xi, r in zip(x, rungs):
        ax.text(
            xi, 1.04, verdicts.get(r, ""), ha="center", fontsize=8,
            color="C3", fontweight="bold",
        )
    return _save(fig, out)


def fig_normalization_gap(out: Path, gap: dict[str, dict[str, float]]) -> Path:
    """Transductive vs inductive normalisation.

    `unsupervised_per_machine` standardises the held-out pump by its own unlabelled
    windows; `train_pooled` never touches it. The gap measures how much of the LOMO
    score depends on having seen the target pump's operating distribution at all —
    which is exactly the question "can you commission a new pump without retraining?"
    """
    models = sorted(set().union(*[set(d) for d in gap.values()]))
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = {
        "unsupervised_per_machine": "per-machine (transductive)",
        "train_pooled": "train-pooled (inductive)",
    }
    for i, (strategy, scores) in enumerate(sorted(gap.items())):
        ax.bar(
            x + (i - 0.5) * w,
            [scores.get(m, 0.0) for m in models],
            w,
            label=labels.get(strategy, strategy),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Macro-F1 (LOMO)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Normalisation strategy gap — does adaptation need the target pump?")
    ax.legend(fontsize=8)
    return _save(fig, out)


def fig_tabpfn_latency(out: Path, bench: list[dict]) -> Path:
    """Measured effect of the KV cache and the ensemble size on predict latency.

    Both optimisations were asserted in the design and neither was enabled in the
    code; these are measurements on this machine, not quoted figures.
    """
    modes = sorted({r["fit_mode"] for r in bench})
    ests = sorted({r["n_estimators"] for r in bench})
    x = np.arange(len(ests))
    w = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = {
        "fit_preprocessors": "preprocessor cache only (library default)",
        "fit_with_cache": "transformer KV cache warmed at boot",
    }
    for i, mode in enumerate(modes):
        vals = [
            next(
                (r["predict_latency_s"] for r in bench
                 if r["fit_mode"] == mode and r["n_estimators"] == e),
                0.0,
            )
            for e in ests
        ]
        bars = ax.bar(x + (i - (len(modes) - 1) / 2) * w, vals, w, label=labels.get(mode, mode))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}s", ha="center",
                    va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n_estimators={e}" for e in ests])
    ax.set_ylabel("Predict latency (s)")
    ctx = bench[0] if bench else {}
    ax.set_title(
        "TabPFN v2 inference latency, CPU\n"
        f"context {ctx.get('n_context', '?')}×{ctx.get('n_features', '?')}, "
        f"{ctx.get('n_query', '?')} query rows"
    )
    ax.legend(fontsize=8)
    return _save(fig, out)


def fig_accuracy_vs_latency(out: Path, points: list[dict]) -> Path:
    """Does the expensive model earn its compute?

    Contribution C4. TabPFN has to beat a tuned GBDT by enough to justify orders of
    magnitude more compute; a log axis is used because that is the scale of the gap.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for p in points:
        lat = max(p["latency_s"], 1e-6)
        ax.scatter(lat, p["macro_f1"], s=90)
        ax.annotate(
            p["model"], (lat, p["macro_f1"]),
            textcoords="offset points", xytext=(8, 5), fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Predict latency for the fold (s, log scale)")
    ax.set_ylabel("Macro-F1 (LOMO)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy vs compute — does the expensive model earn it?")
    ax.grid(alpha=0.3)
    return _save(fig, out)


def fig_calibration(out: Path, per_machine: dict, label: str = "model") -> Path:
    """Reliability diagram pooled across LOMO folds.

    TabPFN's selling point is approximating a Bayesian posterior, so its
    probabilities *should* be calibrated. This is where that gets tested rather
    than asserted.
    """
    conf_all, acc_all, ece_all = [], [], []
    for m in per_machine.values():
        bins = m.get("reliability")
        if bins:
            conf_all.append(bins["bin_conf"])
            acc_all.append(bins["bin_acc"])
        if m.get("ece") is not None:
            ece_all.append(m["ece"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect calibration")
    if conf_all:
        conf_arr = np.array(conf_all, dtype=float)
        acc_arr = np.array(acc_all, dtype=float)
        # Bins no fold populated stay empty rather than warning on an all-NaN mean.
        populated = ~np.all(np.isnan(conf_arr), axis=0)
        if populated.any():
            conf = np.nanmean(conf_arr[:, populated], axis=0)
            acc = np.nanmean(acc_arr[:, populated], axis=0)
            ok = ~(np.isnan(conf) | np.isnan(acc))
            ax.plot(conf[ok], acc[ok], "o-", color="C0")
    ece_txt = f"ECE={np.mean(ece_all):.3f}" if ece_all else "ECE n/a"
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability diagram — {label}\n{ece_txt} (mean over LOMO folds)")
    ax.legend(fontsize=8)
    return _save(fig, out)


def make_all_core_figures(outdir: Path | str) -> list[Path]:
    outdir = Path(outdir)
    paths = [
        fig_cavitation_nonmonotonic(outdir / "A3_cavitation_nonmonotonic.png"),
        fig_trip_false_alarm(outdir / "C2_trip_false_alarm.png"),
        fig_trip_operating_points(outdir / "C7_trip_operating_points.png"),
        fig_energy_battery_life(outdir / "E4_battery_vs_runtime.png"),
        fig_energy_breakdown(outdir / "E3_energy_breakdown.png"),
    ]
    return paths
