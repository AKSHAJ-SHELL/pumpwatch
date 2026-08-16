"""Unit tests for physics module invariants."""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.physics import (
    BearingGeometry,
    DryRunCurrentParams,
    bearing_frequencies_hz,
    cavitation_band_energy,
    cavitation_broadband_intensity,
    dry_run_current,
    dry_run_vpf_amplitude,
    impeller_sideband_amplitudes,
    npsha_m,
    shaft_frequency_hz,
    vane_pass_frequency_hz,
)


def test_shaft_and_vpf_worked_examples():
    # 4-pole ~1440 rpm → 24 Hz; 6-vane → VPF 144 Hz
    assert shaft_frequency_hz(1440) == pytest.approx(24.0)
    assert vane_pass_frequency_hz(1440, 6) == pytest.approx(144.0)
    # 2-pole ~2880 → 48 Hz; 6-vane → 288 Hz
    assert vane_pass_frequency_hz(2880, 6) == pytest.approx(288.0)


def test_bearing_freqs_positive_and_ordered():
    geo = BearingGeometry(n_balls=8, ball_diameter_mm=7.0, pitch_diameter_mm=35.0)
    f = bearing_frequencies_hz(1440, geo)
    assert f["BPFI"] > f["BPFO"] > f["FTF"] > 0
    assert f["BSF"] > 0


def test_cavitation_non_monotonic_peak_and_fall():
    xs = np.linspace(0, 1, 101)
    ys = np.array([cavitation_broadband_intensity(s) for s in xs])
    peak_idx = int(np.argmax(ys))
    # Peak should be interior, not at severity=1
    assert 0.2 < xs[peak_idx] < 0.6
    assert ys[-1] < ys[peak_idx]
    assert ys[0] == pytest.approx(0.0)
    # Severe (1.0) ranks below developed (~0.4)
    assert cavitation_broadband_intensity(1.0) < cavitation_broadband_intensity(0.4)


def test_cavitation_band_energy_increases_then_falls():
    mild = cavitation_band_energy(0.2)
    peak = cavitation_band_energy(0.4)
    choked = cavitation_band_energy(1.0)
    assert peak > mild
    assert choked < peak


def test_dry_run_current_drops_30_to_60_percent():
    t = np.linspace(0, 5, 500)
    params = DryRunCurrentParams(rated_current_a=10.0, dry_run_fraction=0.45, noise_std_fraction=0.0)
    i = dry_run_current(t, onset_s=1.0, params=params, condition="dry_run")
    # After transition, settle near 45% of rated
    settled = i[t > 3.0].mean()
    assert 3.0 < settled < 6.0
    # Pre-onset near rated
    assert i[t < 0.5].mean() == pytest.approx(10.0, abs=0.1)


def test_closed_valve_drops_less_than_dry_run():
    t = np.linspace(0, 5, 500)
    params = DryRunCurrentParams(noise_std_fraction=0.0)
    i_dry = dry_run_current(t, 1.0, params, "dry_run")
    i_cv = dry_run_current(t, 1.0, params, "closed_valve")
    assert i_cv[t > 3].mean() > i_dry[t > 3].mean()


def test_vpf_collapses_under_dry_run():
    t = np.linspace(0, 5, 500)
    amp = dry_run_vpf_amplitude(t, onset_s=1.0, healthy_amp=1.0, collapse_fraction=0.05)
    assert amp[t < 0.5].mean() == pytest.approx(1.0)
    assert amp[t > 3].mean() < 0.15


def test_impeller_sidebands_grow_with_damage():
    healthy = impeller_sideband_amplitudes(0.0)
    damaged = impeller_sideband_amplitudes(1.0)
    assert healthy["VPF_minus_1x"] == pytest.approx(0.0)
    assert damaged["VPF_minus_1x"] > 0.3
    assert damaged["VPF"] < healthy["VPF"]


def test_npsha_basic():
    # Atmospheric suction, water at ~20C vapour pressure ~2.3 kPa
    h = npsha_m(101325.0, 2339.0, 998.0, suction_velocity_m_s=1.0)
    assert 8.0 < h < 12.0


def test_cavitation_severity_derived_from_npsh_margin():
    """Severity is a derived hydraulic quantity, not a free knob."""
    from pumpwatch.physics import cavitation_severity_from_npsh

    comfortable = cavitation_severity_from_npsh(npsha=6.0, npshr=3.0)
    marginal = cavitation_severity_from_npsh(npsha=3.0, npshr=3.0)
    collapsed = cavitation_severity_from_npsh(npsha=0.0, npshr=3.0)
    assert comfortable == 0.0
    assert 0.0 < marginal < 1.0
    assert collapsed == 1.0
    # Inception begins above NPSHr (the 3% head-drop point), which is the argument
    # for vibration-based detection over hydraulic monitoring.
    assert cavitation_severity_from_npsh(npsha=4.0, npshr=3.0) > 0.0


def test_npsh_severity_rejects_nonpositive_npshr():
    from pumpwatch.physics import cavitation_severity_from_npsh

    with pytest.raises(ValueError):
        cavitation_severity_from_npsh(npsha=1.0, npshr=0.0)
