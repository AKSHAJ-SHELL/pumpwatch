"""Physics-based synthetic signal generator for gate/trip validation.

Generates vibration + current time series with known ground-truth labels so
bugs are in the code, not the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from pumpwatch.physics import (
    BearingGeometry,
    DryRunCurrentParams,
    bearing_frequencies_hz,
    cavitation_broadband_intensity,
    dry_run_current,
    dry_run_vpf_amplitude,
    impeller_sideband_amplitudes,
    shaft_frequency_hz,
    vane_pass_frequency_hz,
)


class Condition(str, Enum):
    HEALTHY = "healthy"
    DRY_RUN = "dry_run"
    CLOSED_VALVE = "closed_valve"
    CAVITATION = "cavitation"
    IMPELLER_DAMAGE = "impeller_damage"
    BEARING_OUTER = "bearing_outer"
    BEARING_INNER = "bearing_inner"
    UNBALANCE = "unbalance"
    MISALIGNMENT = "misalignment"
    LOOSENESS = "looseness"


@dataclass
class PumpMeta:
    """Per-pump metadata. Optional fields may be unknown in the field."""

    pump_id: str = "synth_0"
    rpm: float = 1440.0
    n_vanes: Optional[int] = 6
    rated_current_a: float = 10.0
    bearing: Optional[BearingGeometry] = field(
        default_factory=lambda: BearingGeometry(
            n_balls=8, ball_diameter_mm=7.0, pitch_diameter_mm=35.0
        )
    )
    voltage_available: bool = False


@dataclass
class SynthConfig:
    fs_hi: float = 26700.0
    fs_lo: float = 1670.0
    duration_s: float = 2.5
    seed: int = 0
    noise_std: float = 0.05


@dataclass
class SynthRecord:
    """One synthetic acquisition window."""

    t: np.ndarray
    vibration: np.ndarray
    current_rms: np.ndarray  # slow envelope / per-sample RMS proxy
    current_waveform: np.ndarray
    condition: Condition
    severity: float
    meta: PumpMeta
    fs: float
    onset_s: float = 0.0
    labels: dict = field(default_factory=dict)


def _tone(t: np.ndarray, freq: float, amp: float, phase: float = 0.0) -> np.ndarray:
    return amp * np.sin(2.0 * np.pi * freq * t + phase)


def generate_vibration(
    t: np.ndarray,
    meta: PumpMeta,
    condition: Condition,
    severity: float = 0.5,
    onset_s: float = 0.0,
    noise_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Synthesize an accelerometer signal for the given condition."""
    rng = rng or np.random.default_rng(0)
    f1 = shaft_frequency_hz(meta.rpm)
    n_vanes = meta.n_vanes or 6
    vpf = vane_pass_frequency_hz(meta.rpm, n_vanes)

    # Base: 1× + VPF + harmonics (healthy)
    vib = (
        _tone(t, f1, 0.4)
        + _tone(t, 2 * f1, 0.12)
        + _tone(t, vpf, 0.35)
        + _tone(t, 2 * vpf, 0.08)
    )

    if condition == Condition.HEALTHY:
        pass

    elif condition == Condition.DRY_RUN:
        vpf_amp = dry_run_vpf_amplitude(t, onset_s)
        # Rebuild without fixed VPF; apply collapsing VPF amp
        vib = _tone(t, f1, 0.4) + _tone(t, 2 * f1, 0.12)
        vib = vib + vpf_amp * np.sin(2.0 * np.pi * vpf * t)
        # Broadband decreases under dry run
        vib = vib * 0.7
        # Mild HF friction after onset
        hf = np.where(t >= onset_s, 0.08 * severity * rng.standard_normal(t.shape), 0.0)
        vib = vib + hf

    elif condition == Condition.CLOSED_VALVE:
        # Vibration mildly reduced; VPF still present (fluid still there, just no flow)
        vib = vib * (1.0 - 0.15 * severity)

    elif condition == Condition.CAVITATION:
        intensity = cavitation_broadband_intensity(severity)
        # Broadband noise floor rise in 1–6 kHz (sampled via filtered noise proxy)
        bb = intensity * 0.6 * rng.standard_normal(t.shape)
        # Simple band emphasis via differencing (HF boost)
        bb_hf = np.diff(bb, prepend=bb[0])
        vib = vib + bb + 0.5 * bb_hf
        # Non-monotonic: already encoded in intensity

    elif condition == Condition.IMPELLER_DAMAGE:
        sb = impeller_sideband_amplitudes(severity, vpf_amp=0.35)
        vib = (
            _tone(t, f1, 0.4)
            + _tone(t, 2 * f1, 0.12)
            + _tone(t, vpf, sb["VPF"])
            + _tone(t, vpf - f1, sb["VPF_minus_1x"])
            + _tone(t, vpf + f1, sb["VPF_plus_1x"])
        )

    elif condition in (Condition.BEARING_OUTER, Condition.BEARING_INNER):
        if meta.bearing is None:
            raise ValueError("bearing geometry required for bearing faults")
        freqs = bearing_frequencies_hz(meta.rpm, meta.bearing)
        fault_f = freqs["BPFO"] if condition == Condition.BEARING_OUTER else freqs["BPFI"]
        # Impulse train modulated onto HF carrier
        carrier = 4000.0
        impulses = severity * 0.5 * (np.sin(2.0 * np.pi * fault_f * t) > 0.95).astype(float)
        env = impulses * np.sin(2.0 * np.pi * carrier * t)
        vib = vib + env

    elif condition == Condition.UNBALANCE:
        vib = vib + _tone(t, f1, 0.8 * severity)

    elif condition == Condition.MISALIGNMENT:
        vib = vib + _tone(t, 2 * f1, 0.7 * severity) + _tone(t, f1, 0.2 * severity)

    elif condition == Condition.LOOSENESS:
        # Half-order + harmonics
        vib = (
            vib
            + _tone(t, 0.5 * f1, 0.4 * severity)
            + _tone(t, 1.5 * f1, 0.2 * severity)
            + _tone(t, 3 * f1, 0.15 * severity)
        )

    else:
        raise ValueError(f"unhandled condition: {condition}")

    vib = vib + noise_std * rng.standard_normal(t.shape)
    return vib.astype(np.float64)


def generate_current_waveform(
    t: np.ndarray,
    meta: PumpMeta,
    condition: Condition,
    severity: float = 0.5,
    onset_s: float = 0.0,
    line_freq_hz: float = 50.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (waveform, instantaneous_rms_proxy).

    RMS proxy is a sliding envelope matching the dry-run / load model.
    """
    rng = rng or np.random.default_rng(0)
    params = DryRunCurrentParams(rated_current_a=meta.rated_current_a)

    if condition == Condition.DRY_RUN:
        cond_key = "dry_run"
    elif condition == Condition.CLOSED_VALVE:
        cond_key = "closed_valve"
    else:
        cond_key = "healthy"

    # For cavitation etc., current shifts mildly with load
    rms_traj = dry_run_current(t, onset_s, params, condition=cond_key, rng=rng)
    if condition == Condition.CAVITATION:
        # Mild current rise under developing cavitation, fall when choked
        from pumpwatch.physics import cavitation_broadband_intensity

        rms_traj = rms_traj * (1.0 + 0.05 * severity - 0.08 * cavitation_broadband_intensity(severity))

    phase = 2.0 * np.pi * line_freq_hz * t
    waveform = rms_traj * np.sqrt(2.0) * np.sin(phase)
    # Add slight noise
    waveform = waveform + 0.01 * meta.rated_current_a * rng.standard_normal(t.shape)
    return waveform.astype(np.float64), rms_traj.astype(np.float64)


def generate_record(
    condition: Condition = Condition.HEALTHY,
    severity: float = 0.5,
    onset_s: float = 0.5,
    meta: Optional[PumpMeta] = None,
    config: Optional[SynthConfig] = None,
    rate: str = "hi",
) -> SynthRecord:
    """Generate one labelled synthetic window."""
    meta = meta or PumpMeta()
    config = config or SynthConfig()
    rng = np.random.default_rng(config.seed)
    fs = config.fs_hi if rate == "hi" else config.fs_lo
    n = int(config.duration_s * fs)
    t = np.arange(n) / fs

    vib = generate_vibration(
        t, meta, condition, severity=severity, onset_s=onset_s, noise_std=config.noise_std, rng=rng
    )
    i_wave, i_rms = generate_current_waveform(
        t, meta, condition, severity=severity, onset_s=onset_s, rng=rng
    )

    return SynthRecord(
        t=t,
        vibration=vib,
        current_rms=i_rms,
        current_waveform=i_wave,
        condition=condition,
        severity=severity,
        meta=meta,
        fs=fs,
        onset_s=onset_s,
        labels={
            "condition": condition.value,
            "severity": severity,
            "pump_id": meta.pump_id,
            "rpm": meta.rpm,
        },
    )


def generate_dataset(
    n_per_class: int = 20,
    conditions: Optional[list[Condition]] = None,
    n_pumps: int = 2,
    seed: int = 42,
    rate: str = "lo",
) -> list[SynthRecord]:
    """Generate a multi-pump, multi-condition synthetic dataset."""
    conditions = conditions or [
        Condition.HEALTHY,
        Condition.DRY_RUN,
        Condition.CAVITATION,
        Condition.IMPELLER_DAMAGE,
        Condition.BEARING_OUTER,
        Condition.UNBALANCE,
        Condition.MISALIGNMENT,
    ]
    records: list[SynthRecord] = []
    rng = np.random.default_rng(seed)
    for p in range(n_pumps):
        rpm = 1440.0 if p % 2 == 0 else 2950.0
        meta = PumpMeta(pump_id=f"synth_{p}", rpm=rpm, rated_current_a=8.0 + p)
        for cond in conditions:
            for i in range(n_per_class):
                sev = float(rng.uniform(0.2, 0.9)) if cond != Condition.HEALTHY else 0.0
                onset = float(rng.uniform(0.2, 0.8)) if cond in (
                    Condition.DRY_RUN,
                    Condition.CLOSED_VALVE,
                ) else 0.0
                cfg = SynthConfig(seed=int(rng.integers(0, 1_000_000)), duration_s=2.5)
                records.append(
                    generate_record(
                        condition=cond,
                        severity=sev,
                        onset_s=onset,
                        meta=meta,
                        config=cfg,
                        rate=rate,
                    )
                )
    return records
