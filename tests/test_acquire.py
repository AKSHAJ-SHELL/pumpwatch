"""Tests for the dual-rate acquisition model."""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.node.acquire import (
    AcquisitionPlan,
    acquire_dual_rate,
    aliased_frequency,
    decimate_signal,
)


def test_plan_resolutions_match_the_design():
    plan = AcquisitionPlan()
    assert plan.fs_lo == pytest.approx(26_700 / 16)
    # 0.15 s burst with ~6.5 Hz bins; 2.45 s window with ~0.41 Hz bins.
    assert plan.duration_hi_s == pytest.approx(0.153, abs=0.01)
    assert plan.bin_hz_hi == pytest.approx(6.5, abs=0.2)
    assert plan.duration_lo_s == pytest.approx(2.45, abs=0.05)
    assert plan.bin_hz_lo == pytest.approx(0.41, abs=0.02)


def test_high_rate_alone_cannot_resolve_sidebands():
    """The reason two rates are needed rather than one."""
    plan = AcquisitionPlan()
    hi_only = AcquisitionPlan(decimation=1)
    assert plan.resolves_sidebands(rpm=1470.0)
    assert not hi_only.resolves_sidebands(rpm=1470.0)


def test_high_rate_reaches_the_diagnostic_bands():
    plan = AcquisitionPlan()
    assert plan.reaches(6_000.0)  # cavitation band
    assert plan.reaches(12_000.0)  # bearing envelope carrier
    assert not plan.reaches(20_000.0)


def test_naive_subsampling_aliases_the_bearing_carrier():
    """Documents precisely why decimate_signal must not be x[::factor].

    A 4 kHz bearing carrier sampled at 1.67 kSPS folds down onto the shaft orders.
    """
    fs_lo = 26_700 / 16
    assert aliased_frequency(4000.0, fs_lo) == pytest.approx(663.0, abs=2.0)


def test_decimation_removes_out_of_band_energy():
    fs = 26_700.0
    t = np.arange(int(fs * 0.5)) / fs
    # In-band tone we want to keep, plus an out-of-band tone that would alias.
    x = np.sin(2 * np.pi * 120.0 * t) + np.sin(2 * np.pi * 4000.0 * t)

    decimated = decimate_signal(x, 16)
    naive = x[::16]
    fs_lo = fs / 16

    def power_near(sig, f0, bw=8.0):
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
        f = np.fft.rfftfreq(len(sig), d=1.0 / fs_lo)
        m = (f >= f0 - bw) & (f <= f0 + bw)
        return float(spec[m].sum())

    alias_f = aliased_frequency(4000.0, fs_lo)
    # Both keep the wanted tone.
    assert power_near(decimated, 120.0) > 0
    # Only the naive version carries the aliased image.
    assert power_near(naive, alias_f) > 100 * power_near(decimated, alias_f)


def test_acquire_dual_rate_returns_both_windows():
    plan = AcquisitionPlan()
    t = np.arange(int(plan.fs_hi * 1.0)) / plan.fs_hi
    x = np.sin(2 * np.pi * 147.0 * t)
    win = acquire_dual_rate(x, plan.fs_hi, plan)
    assert len(win.hi) == plan.n_hi
    assert win.fs_lo == pytest.approx(plan.fs_lo)
    assert len(win.lo) == pytest.approx(len(x) / plan.decimation, rel=0.02)


def test_acquire_rejects_mismatched_rate():
    with pytest.raises(ValueError, match="does not match"):
        acquire_dual_rate(np.zeros(10_000), 8000.0, AcquisitionPlan())


def test_decimate_rejects_too_short_signal():
    with pytest.raises(ValueError, match="too short"):
        decimate_signal(np.zeros(50), 16)
