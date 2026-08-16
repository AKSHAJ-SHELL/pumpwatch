"""Shaft speed estimation from spectrum — metadata-free field operation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SpeedEstimate:
    f_shaft_hz: float
    confidence: float  # 0–1
    method: str


def estimate_shaft_frequency(
    signal: np.ndarray,
    fs: float,
    rpm_hint: Optional[float] = None,
    search_hz: tuple[float, float] = (10.0, 60.0),
) -> SpeedEstimate:
    """Estimate 1× from the vibration (or current) spectrum peak.

    If rpm_hint is given, search in a ±15% window around it.
    """
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n < 64:
        raise ValueError("signal too short for speed estimation")
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if rpm_hint is not None:
        f0 = rpm_hint / 60.0
        lo, hi = 0.85 * f0, 1.15 * f0
    else:
        lo, hi = search_hz

    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return SpeedEstimate(f_shaft_hz=float("nan"), confidence=0.0, method="failed")

    band = spec[mask]
    band_f = freqs[mask]
    peak_idx = int(np.argmax(band))
    f_est = float(band_f[peak_idx])

    # Confidence: peak prominence vs median in band
    med = float(np.median(band)) + 1e-12
    prominence = float(band[peak_idx]) / med
    confidence = float(np.clip((prominence - 1.0) / 10.0, 0.0, 1.0))

    return SpeedEstimate(f_shaft_hz=f_est, confidence=confidence, method="spectrum_peak")


def rpm_from_hz(f_shaft_hz: float) -> float:
    return f_shaft_hz * 60.0
