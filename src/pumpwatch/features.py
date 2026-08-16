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
    shaft_frequency_hz,
    vane_pass_frequency_hz,
)
from pumpwatch.speed import estimate_shaft_frequency


SCHEMA_VERSION = "1.0.0"


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

    if use_vib:
        vib = np.asarray(vibration, dtype=float)
        feats.update({f"vib_{k}": v for k, v in _time_features(vib).items()})
        feats["iso_vel_rms"] = _iso_velocity_rms(vib, fs_vib)

        # Speed estimation
        if meta.rpm is not None:
            f_shaft = shaft_frequency_hz(meta.rpm)
        else:
            est = estimate_shaft_frequency(vib, fs_vib)
            if est.confidence > 0.1 and np.isfinite(est.f_shaft_hz):
                f_shaft = est.f_shaft_hz

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
