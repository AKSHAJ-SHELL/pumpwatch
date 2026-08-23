"""Node energy model: event-triggered (primary) vs fixed-schedule (falsified).

The old 15-min fixed schedule is computed only as a comparison that fails the
dry-run response requirement. Primary model: wake-on-start, continuous CUSUM
while pump runs, gated LoRa TX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pumpwatch import duty as duty_module
from pumpwatch.node.airtime import (
    LoRaConfig,
    airtime_s,
    feature_vector_payload_bytes,
    heartbeat_payload_bytes,
)


@dataclass(frozen=True)
class CurrentDraw:
    """mA draws for each phase."""

    sample: float = 2.6
    compute: float = 1.5
    lora_tx: float = 45.0  # +14 dBm
    lora_rx: float = 4.2
    sleep: float = 0.010  # Stop2 + SX1262 sleep ≈ 10 µA
    wake_sensor: float = 0.0014  # ADXL372 instant-on 1.4 µA
    cusum_active: float = 1.8  # continuous light sampling while pump runs


@dataclass(frozen=True)
class PhaseTiming:
    sample_s: float = 2.0
    compute_s: float = 0.5
    rx_window_s: float = 0.2


@dataclass
class EnergyResult:
    mAh_per_day: float
    battery_years: float
    tx_fraction: float
    transmissions_per_day: float
    model: str
    notes: str = ""
    # Per-phase mAh/day. Previously these were computed and discarded, and the
    # breakdown figure re-derived the whole calculation by hand — two copies of the
    # same arithmetic that could silently disagree.
    breakdown_mAh: dict = field(default_factory=dict)


def _charge_mAs(current_mA: float, duration_s: float) -> float:
    return current_mA * duration_s


def fixed_schedule_energy(
    interval_s: float = 900.0,
    n_features: int = 30,
    escalate_rate: float = 1.0,
    lora: Optional[LoRaConfig] = None,
    currents: Optional[CurrentDraw] = None,
    timing: Optional[PhaseTiming] = None,
    usable_battery_mAh: float = 2400.0,
) -> EnergyResult:
    """Falsified comparison: fixed duty cycle (the design that cannot meet §0.3)."""
    lora = lora or LoRaConfig(sf=9)
    currents = currents or CurrentDraw()
    timing = timing or PhaseTiming()

    payload = feature_vector_payload_bytes(n_features)
    hb = heartbeat_payload_bytes()
    t_tx_full = airtime_s(payload, lora)
    t_tx_hb = airtime_s(hb, lora)

    cycles_per_day = 86400.0 / interval_s
    # Mix of full TX and heartbeat
    avg_tx_s = escalate_rate * t_tx_full + (1.0 - escalate_rate) * t_tx_hb
    active_s = timing.sample_s + timing.compute_s + avg_tx_s + timing.rx_window_s
    sleep_s = max(interval_s - active_s, 0.0)

    charge = (
        _charge_mAs(currents.sample, timing.sample_s)
        + _charge_mAs(currents.compute, timing.compute_s)
        + _charge_mAs(currents.lora_tx, avg_tx_s)
        + _charge_mAs(currents.lora_rx, timing.rx_window_s)
        + _charge_mAs(currents.sleep, sleep_s)
    )
    mAh_day = (charge / 3600.0) * cycles_per_day
    tx_charge = _charge_mAs(currents.lora_tx, avg_tx_s) * cycles_per_day
    total_charge = charge * cycles_per_day
    years = (usable_battery_mAh / mAh_day) / 365.25 if mAh_day > 0 else float("inf")
    return EnergyResult(
        mAh_per_day=mAh_day,
        battery_years=years,
        tx_fraction=tx_charge / total_charge if total_charge > 0 else 0.0,
        transmissions_per_day=cycles_per_day * escalate_rate,
        model="fixed_schedule",
        notes=f"interval={interval_s}s; FAILS dry-run sub-minute requirement",
    )


def event_triggered_energy(
    pump_runtime_hours_per_day: float = duty_module.DEFAULT_RUNTIME_HOURS_PER_DAY,
    n_features: int = 30,
    escalations_per_runtime_hour: float = 2.0,
    heartbeats_per_runtime_hour: float = 6.0,
    lora: Optional[LoRaConfig] = None,
    currents: Optional[CurrentDraw] = None,
    timing: Optional[PhaseTiming] = None,
    usable_battery_mAh: float = 2400.0,
    # The *decision* cadence: every feature window the gate may escalate is one
    # potential gateway decision, so this is the same rate the alarm budget divides by
    # (see pumpwatch.duty). The denser commissioning burst is ignored here on purpose —
    # it lasts about three days against a multi-year battery life, so it does not move
    # the budget, and pretending to model it would imply a precision this does not have.
    feature_compute_per_runtime_hour: float = (
        duty_module.DEFAULT_DECISION_WINDOWS_PER_RUNTIME_HOUR
    ),
    escalation_rate: Optional[float] = None,
) -> EnergyResult:
    """Primary model: wake on pump start; CUSUM continuous while running; gated TX.

    While pump is off: ADXL372 / CT-threshold wake at ~1.4 µA.
    While pump is on: light continuous sampling for CUSUM (~1.8 mA) plus occasional
    full feature windows and LoRa escalations/heartbeats.

    If ``escalation_rate`` is given, the number of transmissions is derived from it
    rather than assumed: every feature window the MCU gate escalates costs a
    transmission. That is the link between the gate's operating point and battery
    life, and it is the quantitative content of the two-tier architecture claim.
    """
    lora = lora or LoRaConfig(sf=9)
    currents = currents or CurrentDraw()
    timing = timing or PhaseTiming()

    runtime_s = pump_runtime_hours_per_day * 3600.0
    off_s = 86400.0 - runtime_s
    if off_s < 0:
        raise ValueError("pump_runtime_hours_per_day cannot exceed 24")

    payload = feature_vector_payload_bytes(n_features)
    hb = heartbeat_payload_bytes()
    t_tx_full = airtime_s(payload, lora)
    t_tx_hb = airtime_s(hb, lora)

    n_feat = feature_compute_per_runtime_hour * pump_runtime_hours_per_day
    if escalation_rate is not None:
        # Every escalated feature window is one uplink.
        n_esc = n_feat * float(np.clip(escalation_rate, 0.0, 1.0))
    else:
        n_esc = escalations_per_runtime_hour * pump_runtime_hours_per_day
    n_hb = heartbeats_per_runtime_hour * pump_runtime_hours_per_day

    # Charge while running
    charge_cusum = _charge_mAs(currents.cusum_active, runtime_s)
    charge_feat = n_feat * (
        _charge_mAs(currents.sample, timing.sample_s)
        + _charge_mAs(currents.compute, timing.compute_s)
    )
    charge_tx = n_esc * _charge_mAs(currents.lora_tx, t_tx_full) + n_hb * _charge_mAs(
        currents.lora_tx, t_tx_hb
    )
    charge_rx = (n_esc + n_hb) * _charge_mAs(currents.lora_rx, timing.rx_window_s)

    # Off: wake sensor only
    charge_off = _charge_mAs(currents.wake_sensor, off_s)

    parts = {
        "CUSUM active": charge_cusum,
        "Feature windows": charge_feat,
        "LoRa TX": charge_tx,
        "LoRa RX": charge_rx,
        "Wake (off)": charge_off,
    }
    total_charge_mAs = sum(parts.values())
    mAh_day = total_charge_mAs / 3600.0
    years = (usable_battery_mAh / mAh_day) / 365.25 if mAh_day > 0 else float("inf")
    return EnergyResult(
        mAh_per_day=mAh_day,
        battery_years=years,
        tx_fraction=charge_tx / total_charge_mAs if total_charge_mAs > 0 else 0.0,
        transmissions_per_day=n_esc,
        model="event_triggered",
        notes=f"runtime={pump_runtime_hours_per_day}h/day; continuous CUSUM while running",
        breakdown_mAh={k: v / 3600.0 for k, v in parts.items()},
    )
