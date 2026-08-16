"""Schema-versioned, profile-aware feature extractor.

Features degrade gracefully when per-pump metadata (vane count, bearing
geometry) is absent. Power-factor features exist only when voltage_available.
Vector is not hardcoded to 42 — the schema declares what is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as sps

from pumpwatch.physics import (
    BearingGeometry,
    bearing_frequencies_hz,
    rotor_bar_sidebands_hz,
    shaft_frequency_hz,
    vane_pass_frequency_hz,
)
from pumpwatch.speed import estimate_shaft_frequency


# 1.1.0 adds MCSA + current-trajectory features, so ct_only is no longer two scalars.
SCHEMA_VERSION = "1.1.0"


@dataclass
class FeatureMeta:
    """Optional per-pump metadata. Missing fields → subset of features."""

    rpm: Optional[float] = None
    n_vanes: Optional[int] = None
    bearing: Optional[BearingGeometry] = None
    rated_current_a: Optional[float] = None
    voltage_available: bool = False
    profile: str = "full"  # 'full' | 'ct_only'


@dataclass
class FeatureVector:
    names: list[str]
    values: np.ndarray
    schema_version: str = SCHEMA_VERSION
    profile: str = "full"
    f_shaft_hz: Optional[float] = None

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values.tolist()))


def _time_features(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    pp = float(np.ptp(x))
    std = float(np.std(x)) + 1e-12
    mean = float(np.mean(x))
    # Kurtosis / skew (Fisher)
    z = (x - mean) / std
    kurt = float(np.mean(z**4) - 3.0)
    skew = float(np.mean(z**3))
    crest = peak / (rms + 1e-12)
    shape = rms / (float(np.mean(np.abs(x))) + 1e-12)
    impulse = peak / (float(np.mean(np.abs(x))) + 1e-12)
    clearance = peak / (float(np.mean(np.sqrt(np.abs(x))) ** 2) + 1e-12)
    return {
        "rms": rms,
        "peak": peak,
        "pp": pp,
        "std": std,
        "kurtosis": kurt,
        "skew": skew,
        "crest": crest,
        "shape": shape,
        "impulse": impulse,
        "clearance": clearance,
    }


def _spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * w)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spec


def _band_power(freqs: np.ndarray, spec: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.sum(spec[mask]))


def _peak_near(freqs: np.ndarray, spec: np.ndarray, f0: float, bw: float = 1.0) -> float:
    return _band_power(freqs, spec, f0 - bw, f0 + bw)


def _iso_velocity_rms(accel: np.ndarray, fs: float) -> float:
    """Approximate velocity RMS in 10–1000 Hz via integration in frequency domain."""
    freqs, spec_a = _spectrum(accel, fs)
    # Acceleration PSD → velocity: divide by (2πf)^2 for power
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.zeros_like(freqs)
        mask = freqs >= 1.0
        scale[mask] = 1.0 / (2.0 * np.pi * freqs[mask]) ** 2
    spec_v = spec_a * scale
    return float(np.sqrt(_band_power(freqs, spec_v, 10.0, 1000.0) + 1e-30))


def _envelope_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert envelope → spectrum for bearing defect frequencies."""
    analytic = sps.hilbert(x)
    env = np.abs(analytic)
    env = env - np.mean(env)
    return _spectrum(env, fs)


def _current_trajectory_features(i: np.ndarray, rated_a: Optional[float]) -> dict[str, float]:
    """Shape of the current-RMS trajectory over the window.

    A dry-run or load-loss event is a *transition*, not a level, so the slope and
    the start/end ratio carry information the steady-state mean throws away.
    """
    i = np.asarray(i, dtype=float)
    if i.size < 4:
        return {}
    n = i.size
    head = float(np.mean(i[: n // 4]))
    tail = float(np.mean(i[-n // 4 :]))
    denom = abs(head) + 1e-12
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, i, 1)[0]) * n  # change across the whole window
    out = {
        "current_traj_drop_ratio": tail / denom,
        "current_traj_slope": slope / denom,
        "current_traj_cv": float(np.std(i) / (np.mean(np.abs(i)) + 1e-12)),
        "current_traj_min_ratio": float(np.min(i)) / denom,
    }
    if rated_a is not None and rated_a > 0:
        out["current_traj_min_vs_rated"] = float(np.min(i)) / rated_a
    return out


def _mcsa_features(
    wave: np.ndarray,
    fs: float,
    f_shaft: Optional[float],
    meta: "FeatureMeta",
    line_freq_hz: float = 50.0,
) -> dict[str, float]:
    """Motor-current signature analysis.

    Mechanical faults impose periodic torque disturbances that amplitude-modulate
    stator current, placing energy at f_line ± k·f_disturbance. Sideband power is
    reported *relative to the fundamental*, which makes it invariant to load and to
    the CT's absolute scaling — necessary if a reference set is to transfer between
    pumps of different sizes.
    """
    if wave.size < 64 or fs <= 2 * line_freq_hz:
        return {}

    freqs, spec = _spectrum(wave, fs)
    # Resolution-aware half-width: a fixed ±1 Hz window silently misses the peak
    # when the record is short.
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    bw = max(1.5 * df, 0.5)

    fundamental = _peak_near(freqs, spec, line_freq_hz, bw)
    if fundamental <= 0.0:
        return {}
    total = float(np.sum(spec)) + 1e-30

    def rel(f0: float) -> float:
        return _peak_near(freqs, spec, f0, bw) / fundamental

    out: dict[str, float] = {
        "mcsa_fundamental_frac": fundamental / total,
        "mcsa_thd_proxy": sum(rel(k * line_freq_hz) for k in (2, 3, 5)),
    }

    # Broken rotor bar: f_line(1 ± 2s). Slip is unknown at inference, so sweep the
    # plausible band and take the strongest sideband rather than assuming a value.
    rb_band = []
    for slip in np.linspace(0.005, 0.06, 12):
        rb_band.extend(rotor_bar_sidebands_hz(line_freq_hz, float(slip)))
    out["mcsa_rotorbar_max"] = max((rel(f) for f in rb_band), default=0.0)

    if f_shaft is not None and f_shaft > 0:
        # Mechanical fault sidebands mirrored onto the line.
        for k in (1, 2):
            lo, hi = line_freq_hz - k * f_shaft, line_freq_hz + k * f_shaft
            out[f"mcsa_sb_{k}x_lower"] = rel(lo)
            out[f"mcsa_sb_{k}x_upper"] = rel(hi)
            out[f"mcsa_sb_{k}x_sum"] = rel(lo) + rel(hi)
        # Half-order: mechanical looseness.
        out["mcsa_sb_0p5x_sum"] = rel(line_freq_hz - 0.5 * f_shaft) + rel(
            line_freq_hz + 0.5 * f_shaft
        )

        n_vanes = meta.n_vanes
        if n_vanes:
            vpf = n_vanes * f_shaft
            out["mcsa_vpf_sb_sum"] = rel(line_freq_hz - vpf) + rel(line_freq_hz + vpf)
            # VPF ± 1x mirrored onto the line: the impeller-damage discriminator,
            # visible on a CT even when the pump cannot be reached with an accelerometer.
            out["mcsa_vpf_1x_sb_sum"] = sum(
                rel(line_freq_hz + s1 * (vpf + s2 * f_shaft))
                for s1 in (-1.0, 1.0)
                for s2 in (-1.0, 1.0)
            )

        if meta.bearing is not None and meta.rpm is not None:
            bf = bearing_frequencies_hz(meta.rpm, meta.bearing)
            for name in ("BPFO", "BPFI"):
                out[f"mcsa_{name.lower()}_sb_sum"] = rel(line_freq_hz - bf[name]) + rel(
                    line_freq_hz + bf[name]
                )

    # Cavitation raises a stochastic torque-noise floor rather than discrete lines,
    # so the residual after removing the harmonic comb is the informative quantity.
    harmonic_mask = np.zeros_like(freqs, dtype=bool)
    for k in range(1, int(fs / 2 / line_freq_hz) + 1):
        harmonic_mask |= np.abs(freqs - k * line_freq_hz) <= bw
    noise_floor = float(np.sum(spec[~harmonic_mask]))
    out["mcsa_noise_floor_frac"] = noise_floor / total
    out["mcsa_noise_to_fundamental"] = noise_floor / fundamental
    return out


def extract_features(
    vibration: Optional[np.ndarray],
    fs_vib: Optional[float],
    current_rms: Optional[np.ndarray] = None,
    current_waveform: Optional[np.ndarray] = None,
    fs_current: Optional[float] = None,
    seal_temp_c: Optional[float] = None,
    discharge_pressure: Optional[float] = None,
    meta: Optional[FeatureMeta] = None,
) -> FeatureVector:
    """Extract a schema-versioned feature vector for the given profile."""
    meta = meta or FeatureMeta()
    feats: dict[str, float] = {}
    f_shaft: Optional[float] = None

    profile = meta.profile
    use_vib = profile == "full" and vibration is not None and fs_vib is not None

    # Shaft speed is resolved before any branch: the MCSA sideband features need it
    # just as much as the vibration order features do, and on ct_only (submersible)
    # there is no vibration channel to estimate it from. Nameplate rpm first, then
    # whichever signal is actually available.
    if meta.rpm is not None:
        f_shaft = shaft_frequency_hz(meta.rpm)
    else:
        for sig, sig_fs in ((vibration, fs_vib), (current_waveform, fs_current)):
            if sig is None or sig_fs is None:
                continue
            est = estimate_shaft_frequency(np.asarray(sig, dtype=float), sig_fs)
            if est.confidence > 0.1 and np.isfinite(est.f_shaft_hz):
                f_shaft = est.f_shaft_hz
                break

    if use_vib:
        vib = np.asarray(vibration, dtype=float)
        feats.update({f"vib_{k}": v for k, v in _time_features(vib).items()})
        feats["iso_vel_rms"] = _iso_velocity_rms(vib, fs_vib)

        freqs, spec = _spectrum(vib, fs_vib)
        total = float(np.sum(spec)) + 1e-30

        if f_shaft is not None:
            for k, mult in [("0p5x", 0.5), ("1x", 1), ("2x", 2), ("3x", 3), ("4x", 4), ("5x", 5)]:
                feats[f"order_{k}"] = _peak_near(freqs, spec, mult * f_shaft) / total

            if meta.n_vanes is not None:
                vpf = meta.n_vanes * f_shaft
                feats["vpf"] = _peak_near(freqs, spec, vpf) / total
                feats["vpf_2x"] = _peak_near(freqs, spec, 2 * vpf) / total
                feats["vpf_3x"] = _peak_near(freqs, spec, 3 * vpf) / total
                feats["vpf_minus_1x"] = _peak_near(freqs, spec, vpf - f_shaft) / total
                feats["vpf_plus_1x"] = _peak_near(freqs, spec, vpf + f_shaft) / total

            if meta.bearing is not None:
                bf = bearing_frequencies_hz(f_shaft * 60.0, meta.bearing)
                ef, espec = _envelope_spectrum(vib, fs_vib)
                etot = float(np.sum(espec)) + 1e-30
                for name, f0 in bf.items():
                    for h in (1, 2, 3):
                        feats[f"env_{name}_h{h}"] = _peak_near(ef, espec, h * f0) / etot

        # Broadband
        feats["bb_centroid"] = float(np.sum(freqs * spec) / total)
        p = spec / total
        feats["bb_entropy"] = float(-np.sum(p * np.log(p + 1e-30)))
        geo = float(np.exp(np.mean(np.log(spec + 1e-30))))
        feats["bb_flatness"] = geo / (float(np.mean(spec)) + 1e-30)
        for lo, hi, label in [(1000, 2000, "1_2k"), (2000, 4000, "2_4k"), (4000, 6000, "4_6k")]:
            feats[f"band_{label}"] = _band_power(freqs, spec, lo, hi) / total

    # Current features (both profiles)
    if current_rms is not None:
        i = np.asarray(current_rms, dtype=float)
        i_rms = float(np.sqrt(np.mean(i**2))) if i.ndim else float(i)
        # If it's a trajectory, use mean of latter half as steady-state
        if i.size > 1:
            i_rms = float(np.mean(i[len(i) // 2 :]))
        feats["current_rms"] = i_rms
        if meta.rated_current_a is not None and meta.rated_current_a > 0:
            feats["current_rms_ratio"] = i_rms / meta.rated_current_a
        feats.update(_current_trajectory_features(i, meta.rated_current_a))

    # MCSA on the current waveform. Without this the ct_only profile carries two
    # scalars derived from one number and cannot separate fault classes at all —
    # which would make DESIGN §0.3's "honest headline profile" a null result.
    if current_waveform is not None and fs_current is not None:
        feats.update(
            _mcsa_features(
                np.asarray(current_waveform, dtype=float),
                fs_current,
                f_shaft=f_shaft,
                meta=meta,
            )
        )

    if meta.voltage_available and current_waveform is not None and fs_current is not None:
        # Placeholder PF proxy: requires voltage. Only computed when flagged.
        # Without a voltage channel we refuse to invent it.
        feats["power_factor_proxy"] = float("nan")  # caller must supply voltage path
        # Real implementation would correlate v and i; left as NaN sentinel unless
        # voltage series is later added to the API.
        del feats["power_factor_proxy"]

    if seal_temp_c is not None:
        feats["seal_temp_c"] = float(seal_temp_c)

    if discharge_pressure is not None and profile == "full":
        feats["discharge_pressure"] = float(discharge_pressure)

    # ct_only profile: strip any vibration keys that slipped through
    if profile == "ct_only":
        feats = {k: v for k, v in feats.items() if not k.startswith(("vib_", "iso_", "order_", "vpf", "env_", "bb_", "band_"))}

    names = sorted(feats.keys())
    values = np.array([feats[n] for n in names], dtype=np.float64)
    return FeatureVector(
        names=names,
        values=values,
        schema_version=SCHEMA_VERSION,
        profile=profile,
        f_shaft_hz=f_shaft,
    )


def feature_matrix(
    vectors: list[FeatureVector],
    reference_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Stack FeatureVectors into a matrix with aligned columns (missing → 0)."""
    if not vectors:
        return np.zeros((0, 0)), []
    names = reference_names or sorted(set().union(*(set(v.names) for v in vectors)))
    X = np.zeros((len(vectors), len(names)), dtype=np.float64)
    for i, v in enumerate(vectors):
        lookup = dict(zip(v.names, v.values))
        for j, n in enumerate(names):
            X[i, j] = lookup.get(n, 0.0)
    return X, names
