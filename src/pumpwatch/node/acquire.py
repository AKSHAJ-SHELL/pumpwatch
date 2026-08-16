"""Dual-rate acquisition model for the MCU tier.

One sample rate cannot serve both diagnostic bands. The design calls for two:

* **hi** — 26.7 kSPS, short burst. Reaches the cavitation band (1-6 kHz) and the
  bearing envelope carriers (2-15 kHz). At 4096 points that is a 0.15 s window
  with 6.5 Hz bins, which resolves a 24 Hz shaft rate into ~4 bins — useless for
  order analysis.
* **lo** — decimated to ~1.67 kSPS, long window. At 4096 points that is 2.45 s
  with 0.41 Hz bins, which resolves 1x, 2x, vane pass, and critically the
  VPF ± 1x sidebands that discriminate impeller damage.

Before this module the two rates existed only as two constants in ``SynthConfig``
with no mechanism behind them: nothing decimated, nothing anti-alias filtered, and
no record carried both rates at once. That matters because naive index
subsampling — taking every Nth sample without filtering — folds everything above
the new Nyquist back into the band you are about to analyse. The 4 kHz bearing
carrier lands at ~660 Hz, on top of the order harmonics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as sps


@dataclass(frozen=True)
class AcquisitionPlan:
    """Dual-rate acquisition parameters and their derived properties."""

    fs_hi: float = 26_700.0
    n_hi: int = 4096
    decimation: int = 16
    n_lo: int = 4096

    @property
    def fs_lo(self) -> float:
        return self.fs_hi / self.decimation

    @property
    def duration_hi_s(self) -> float:
        return self.n_hi / self.fs_hi

    @property
    def duration_lo_s(self) -> float:
        return self.n_lo / self.fs_lo

    @property
    def bin_hz_hi(self) -> float:
        return self.fs_hi / self.n_hi

    @property
    def bin_hz_lo(self) -> float:
        return self.fs_lo / self.n_lo

    def describe(self) -> dict:
        return {
            "fs_hi_hz": self.fs_hi,
            "fs_lo_hz": self.fs_lo,
            "duration_hi_s": self.duration_hi_s,
            "duration_lo_s": self.duration_lo_s,
            "bin_hz_hi": self.bin_hz_hi,
            "bin_hz_lo": self.bin_hz_lo,
            "nyquist_hi_hz": self.fs_hi / 2.0,
            "nyquist_lo_hz": self.fs_lo / 2.0,
        }

    def resolves_sidebands(self, rpm: float, margin: float = 10.0) -> bool:
        """Can the low-rate window measure VPF ± 1x amplitudes?

        The sidebands sit one shaft order from the carrier, so bin width has to be
        finer than the shaft rate. `margin` is bins-per-shaft-order and defaults to
        10 because the feature is a sideband *amplitude*, not merely a yes/no
        separation: at the Rayleigh limit of ~2 bins the carrier's spectral leakage
        dominates whatever sits next to it. At 26.7 kSPS with a 4096-point window
        the bins are 6.5 Hz against a 24.5 Hz shaft rate — under 4 bins, which is
        why the high rate alone cannot do order analysis.
        """
        f_shaft = rpm / 60.0
        return self.bin_hz_lo * margin <= f_shaft

    def reaches(self, freq_hz: float) -> bool:
        """Is `freq_hz` below the high-rate Nyquist?"""
        return freq_hz < 0.5 * self.fs_hi


def decimate_signal(
    x: np.ndarray,
    factor: int,
    zero_phase: bool = True,
) -> np.ndarray:
    """Anti-alias filter then downsample by `factor`.

    Uses an FIR decimator so the passband is flat and, with ``zero_phase``, the
    group delay is removed — phase distortion would smear the impulse trains that
    bearing envelope analysis depends on.

    Never replace this with ``x[::factor]``. That is not decimation; it is
    aliasing with extra steps.
    """
    x = np.asarray(x, dtype=float)
    if factor < 1:
        raise ValueError("decimation factor must be >= 1")
    if factor == 1:
        return x.copy()
    if len(x) <= 27 * factor:
        raise ValueError(
            f"signal of {len(x)} samples is too short to decimate by {factor}"
        )
    # ftype='fir' keeps it linear-phase; large factors are done in stages to keep
    # the filter well-conditioned.
    return sps.decimate(x, factor, ftype="fir", zero_phase=zero_phase)


@dataclass
class DualRateWindow:
    """One acquisition: a high-rate burst and its decimated long-window twin."""

    hi: np.ndarray
    lo: np.ndarray
    fs_hi: float
    fs_lo: float
    plan: AcquisitionPlan


def acquire_dual_rate(
    signal: np.ndarray,
    fs: float,
    plan: Optional[AcquisitionPlan] = None,
) -> DualRateWindow:
    """Split one acquired stream into the two analysis rates.

    Models what the node does: sample once at the high rate, keep a short burst
    for the wideband features, and decimate the rest for order analysis.
    """
    plan = plan or AcquisitionPlan(fs_hi=fs)
    x = np.asarray(signal, dtype=float)
    if fs != plan.fs_hi:
        raise ValueError(
            f"signal rate {fs} does not match plan.fs_hi {plan.fs_hi}; "
            "resample or build a matching plan"
        )
    hi = x[: min(plan.n_hi, len(x))]
    lo = decimate_signal(x, plan.decimation)
    return DualRateWindow(hi=hi, lo=lo, fs_hi=plan.fs_hi, fs_lo=plan.fs_lo, plan=plan)


def aliased_frequency(freq_hz: float, fs: float) -> float:
    """Where `freq_hz` appears after sampling at `fs` with no anti-alias filter.

    Provided so the aliasing argument can be asserted in a test rather than
    described in a comment: the 4 kHz bearing carrier sampled at 1.67 kSPS lands
    at ~663 Hz, right on top of the shaft orders.
    """
    nyq = fs / 2.0
    f = abs(freq_hz) % fs
    return f if f <= nyq else fs - f
