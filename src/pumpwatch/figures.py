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
from pumpwatch.node.trip import TripConfig, evaluate_trip_path
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
    result = evaluate_trip_path(
        n_trials=n_trials,
        seed=seed,
        trip_config=TripConfig(cusum_h=3.5, persistence_n=2),
        duration_s=6.0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Panel A: rates
    labels = ["dry-run\ndetect", "closed-valve\nfalse trip", "healthy\nfalse trip"]
    vals = [
        result.dry_run_detection_rate,
        result.closed_valve_false_trip_rate,
        result.healthy_false_trip_rate,
    ]
    colors = ["C2", "C3", "C1"]
    axes[0].bar(labels, vals, color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Trip path: detection vs confusers")
    # Panel B: delay distribution
    if result.delays_s:
        axes[1].hist(result.delays_s, bins=15, color="C0", edgecolor="white")
        axes[1].axvline(result.dry_run_median_delay_s, color="k", ls="--", label=f"median={result.dry_run_median_delay_s:.2f}s")
        axes[1].legend()
    axes[1].set_xlabel("Detection delay (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Dry-run detection delay")
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


def fig_energy_breakdown(out: Path, runtime_h: float = 3.0) -> Path:
    """Approximate stacked breakdown for event-triggered at given runtime."""
    from pumpwatch.node.airtime import LoRaConfig, airtime_s, feature_vector_payload_bytes, heartbeat_payload_bytes
    from pumpwatch.node.energy import CurrentDraw, PhaseTiming, _charge_mAs

    currents = CurrentDraw()
    timing = PhaseTiming()
    lora = LoRaConfig(sf=9)
    runtime_s = runtime_h * 3600
    off_s = 86400 - runtime_s
    n_esc = 2.0 * runtime_h
    n_hb = 6.0 * runtime_h
    n_feat = 12.0 * runtime_h
    t_tx = airtime_s(feature_vector_payload_bytes(30), lora)
    t_hb = airtime_s(heartbeat_payload_bytes(), lora)

    parts = {
        "CUSUM active": _charge_mAs(currents.cusum_active, runtime_s),
        "Feature windows": n_feat * (
            _charge_mAs(currents.sample, timing.sample_s)
            + _charge_mAs(currents.compute, timing.compute_s)
        ),
        "LoRa TX": n_esc * _charge_mAs(currents.lora_tx, t_tx)
        + n_hb * _charge_mAs(currents.lora_tx, t_hb),
        "LoRa RX": (n_esc + n_hb) * _charge_mAs(currents.lora_rx, timing.rx_window_s),
        "Wake (off)": _charge_mAs(currents.wake_sensor, off_s),
    }
    total = sum(parts.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(parts.keys())
    vals = [parts[k] / 3600 for k in labels]  # mAh
    ax.barh(labels, vals, color=["C0", "C1", "C3", "C4", "C2"])
    ax.set_xlabel("mAh / day")
    ax.set_title(f"Event-triggered energy breakdown ({runtime_h} h/day runtime)")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.2f} ({100*parts[labels[i]]/total:.0f}%)", va="center", fontsize=8)
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


def fig_lomo_per_machine(out: Path, per_machine: dict[str, float]) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(per_machine.keys())
    vals = [per_machine[n] for n in names]
    ax.bar(names, vals, color="C0")
    ax.set_ylabel("Macro-F1")
    ax.set_title("LOMO per-machine results (n machines = thesis unit)")
    ax.set_ylim(0, 1.05)
    ax.axhline(np.mean(vals), ls="--", color="k", label=f"mean={np.mean(vals):.2f}")
    ax.legend()
    return _save(fig, out)


def fig_calibration(out: Path, bin_conf: list, bin_acc: list, ece: float, label: str = "model") -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray")
    ax.plot(bin_conf, bin_acc, "o-", label=f"{label} ECE={ece:.3f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    return _save(fig, out)


def make_all_core_figures(outdir: Path | str) -> list[Path]:
    outdir = Path(outdir)
    paths = [
        fig_cavitation_nonmonotonic(outdir / "A3_cavitation_nonmonotonic.png"),
        fig_trip_false_alarm(outdir / "C2_trip_false_alarm.png"),
        fig_energy_battery_life(outdir / "E4_battery_vs_runtime.png"),
        fig_energy_breakdown(outdir / "E3_energy_breakdown.png"),
    ]
    return paths
