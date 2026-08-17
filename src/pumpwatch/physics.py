"""Pump fault physics: frequencies, NPSH, cavitation, dry-run current models.

All formulas are deterministic so synthetic tests have known ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Characteristic frequencies
# ---------------------------------------------------------------------------


def shaft_frequency_hz(rpm: float) -> float:
    """Shaft (1×) frequency in Hz from rotational speed in rpm."""
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")
    return rpm / 60.0


def vane_pass_frequency_hz(rpm: float, n_vanes: int) -> float:
    """VPF = Z × f_shaft."""
    if n_vanes < 1:
        raise ValueError(f"n_vanes must be >= 1, got {n_vanes}")
    return n_vanes * shaft_frequency_hz(rpm)


@dataclass(frozen=True)
class BearingGeometry:
    """Rolling-element bearing geometry for envelope frequency calculation."""

    n_balls: int
    ball_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.n_balls < 1:
            raise ValueError("n_balls must be >= 1")
        if self.ball_diameter_mm <= 0 or self.pitch_diameter_mm <= 0:
            raise ValueError("diameters must be positive")
        if self.ball_diameter_mm >= self.pitch_diameter_mm:
            raise ValueError("ball diameter must be < pitch diameter")


def bearing_frequencies_hz(rpm: float, geo: BearingGeometry) -> dict[str, float]:
    """BPFO, BPFI, BSF, FTF at given shaft speed."""
    f = shaft_frequency_hz(rpm)
    d = geo.ball_diameter_mm
    D = geo.pitch_diameter_mm
    n = geo.n_balls
    cos_a = np.cos(np.deg2rad(geo.contact_angle_deg))
    return {
        "BPFO": 0.5 * n * f * (1.0 - (d / D) * cos_a),
        "BPFI": 0.5 * n * f * (1.0 + (d / D) * cos_a),
        "BSF": 0.5 * (D / d) * f * (1.0 - ((d / D) * cos_a) ** 2),
        "FTF": 0.5 * f * (1.0 - (d / D) * cos_a),
    }


# ---------------------------------------------------------------------------
# Hydraulic / NPSH
# ---------------------------------------------------------------------------


def npsha_m(
    suction_pressure_pa: float,
    vapour_pressure_pa: float,
    density_kg_m3: float,
    suction_velocity_m_s: float = 0.0,
    g: float = 9.80665,
) -> float:
    """NPSH available (metres): (p_s − p_v)/(ρg) + v²/(2g)."""
    if density_kg_m3 <= 0 or g <= 0:
        raise ValueError("density and g must be positive")
    return (suction_pressure_pa - vapour_pressure_pa) / (density_kg_m3 * g) + (
        suction_velocity_m_s**2
    ) / (2.0 * g)


def npsh_margin(npsha: float, npshr: float) -> float:
    """Positive margin means cavitation risk is low."""
    return npsha - npshr


def cavitation_severity_from_npsh(npsha: float, npshr: float) -> float:
    """Map NPSH margin to the [0, 1] cavitation severity the models use.

    Ties the hydraulic state to the vibration model, so severity is a derived
    physical quantity rather than a free knob. Previously ``npsha_m`` and
    ``npsh_margin`` were computed nowhere outside their own unit tests, and every
    synthetic cavitation record took an arbitrary severity — which means the
    generator could not answer the question the sensor suite is meant to answer:
    how far below NPSHr does the indicator start responding?

    Convention: severity 0 at a comfortable margin (>= NPSHr, i.e. ratio >= 1),
    rising to 1 when NPSHa collapses to zero. NPSHr is conventionally defined at
    the 3% head-drop point (NPSH3), which is already well-developed cavitation —
    inception happens *above* it, which is the whole argument for vibration-based
    detection.
    """
    if npshr <= 0:
        raise ValueError("npshr must be positive")
    ratio = max(npsha, 0.0) / npshr
    # Inception around ratio ~1.5, fully developed by ratio ~0.
    return float(np.clip((1.5 - ratio) / 1.5, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Cavitation vibration response (non-monotonic: rise → peak → fall)
# ---------------------------------------------------------------------------


def cavitation_broadband_intensity(severity: float) -> float:
    """Relative broadband vibration intensity vs cavitation severity.

    severity in [0, 1]:
      0     = healthy (no cavitation)
      ~0.4  = peak intensity (developed cavitation, max collapse energy)
      1.0   = choked / vapour-blanketed (damped, intensity falls)

    This encodes the non-monotonicity trap: a monotonic threshold detector
    will rank severe cavitation as mild.
    """
    s = float(np.clip(severity, 0.0, 1.0))
    # Asymmetric peak near 0.4: rise steep, fall gentler
    peak = 0.40
    if s <= peak:
        return (s / peak) ** 1.2
    # Fall from 1.0 at peak toward ~0.25 at severity=1
    t = (s - peak) / (1.0 - peak)
    return 1.0 - 0.75 * (t**1.5)


def cavitation_band_energy(
    severity: float,
    band_hz: tuple[float, float] = (1000.0, 2000.0),
    base_noise: float = 0.05,
) -> float:
    """Band energy proxy for a given frequency band under cavitation.

    Mid-bands (1–2 kHz) are most sensitive; higher bands scale similarly
    but with lower absolute amplitude.
    """
    intensity = cavitation_broadband_intensity(severity)
    lo, hi = band_hz
    # Sensitivity weight: peak around 1–2 kHz (Al-Obaidi)
    centre = 0.5 * (lo + hi)
    sensitivity = np.exp(-0.5 * ((np.log10(max(centre, 1.0)) - np.log10(1500.0)) / 0.4) ** 2)
    return float(base_noise + intensity * sensitivity)


# ---------------------------------------------------------------------------
# Dry-run current model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DryRunCurrentParams:
    """Parameters for the under-current dry-run signature."""

    rated_current_a: float = 10.0
    dry_run_fraction: float = 0.45  # fraction of rated under dry run (30–60%)
    closed_valve_fraction: float = 0.70  # closed discharge also drops current
    noise_std_fraction: float = 0.02
    transition_s: float = 0.5  # how fast current drops after dry onset


def dry_run_current(
    t: np.ndarray,
    onset_s: float,
    params: Optional[DryRunCurrentParams] = None,
    condition: str = "dry_run",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate motor current RMS trajectory through a dry-run (or closed-valve) event.

    Parameters
    ----------
    t : time axis (seconds)
    onset_s : when the fault begins
    condition : 'dry_run' | 'closed_valve' | 'healthy'
    """
    p = params or DryRunCurrentParams()
    rng = rng or np.random.default_rng(0)
    i = np.full_like(t, p.rated_current_a, dtype=float)

    if condition == "healthy":
        target_frac = 1.0
    elif condition == "dry_run":
        target_frac = p.dry_run_fraction
    elif condition == "closed_valve":
        target_frac = p.closed_valve_fraction
    else:
        raise ValueError(f"unknown condition: {condition}")

    if condition != "healthy":
        # Smooth transition after onset
        tau = max(p.transition_s, 1e-3)
        frac = np.where(
            t < onset_s,
            1.0,
            1.0 + (target_frac - 1.0) * (1.0 - np.exp(-(t - onset_s) / tau)),
        )
        i = p.rated_current_a * frac

    noise = rng.normal(0.0, p.noise_std_fraction * p.rated_current_a, size=t.shape)
    return np.maximum(i + noise, 0.0)


def dry_run_vpf_amplitude(
    t: np.ndarray,
    onset_s: float,
    healthy_amp: float = 1.0,
    collapse_fraction: float = 0.05,
    transition_s: float = 0.8,
) -> np.ndarray:
    """VPF amplitude collapses toward near-zero under dry running while 1× persists."""
    tau = max(transition_s, 1e-3)
    amp = np.where(
        t < onset_s,
        healthy_amp,
        healthy_amp
        * (
            collapse_fraction
            + (1.0 - collapse_fraction) * np.exp(-(t - onset_s) / tau)
        ),
    )
    return amp.astype(float)


# ---------------------------------------------------------------------------
# Impeller damage: VPF ± 1× sidebands
# ---------------------------------------------------------------------------


def motor_slip(load_fraction: float, slip_full_load: float = 0.03) -> float:
    """Induction-motor slip, approximately linear in load over the normal range.

    Matters because the broken-rotor-bar signature sits at f_line ± 2·s·f_line, so
    the sideband location moves with load. A detector that assumes a fixed offset
    will miss it on a lightly loaded pump.
    """
    return float(np.clip(load_fraction, 0.0, 1.5)) * slip_full_load


def rotor_bar_sidebands_hz(line_freq_hz: float, slip: float) -> list[float]:
    """Broken-rotor-bar sidebands at f_line·(1 ± 2s)."""
    return sorted(
        f for f in (line_freq_hz * (1.0 - 2.0 * slip), line_freq_hz * (1.0 + 2.0 * slip))
        if f > 0.0
    )


def impeller_sideband_amplitudes(
    damage: float,
    vpf_amp: float = 1.0,
) -> dict[str, float]:
    """Return relative amplitudes at VPF and VPF±1× given impeller damage in [0,1].

    Healthy (damage=0): VPF present, sidebands ~0.
    Damaged: once-per-rev modulation → sidebands at VPF ± f_shaft grow.
    """
    d = float(np.clip(damage, 0.0, 1.0))
    return {
        "VPF": vpf_amp * (1.0 - 0.3 * d),  # slight VPF drop as symmetry breaks
        "VPF_minus_1x": vpf_amp * 0.45 * d,
        "VPF_plus_1x": vpf_amp * 0.45 * d,
    }
