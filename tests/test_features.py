"""Tests for feature extraction and speed estimation."""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.features import FeatureMeta, extract_features, feature_matrix
from pumpwatch.physics import BearingGeometry
from pumpwatch.speed import estimate_shaft_frequency
from pumpwatch.synth import Condition, PumpMeta, SynthConfig, generate_record


def test_speed_estimation_near_true():
    meta = PumpMeta(rpm=1440.0)
    rec = generate_record(
        Condition.HEALTHY,
        meta=meta,
        config=SynthConfig(duration_s=2.5, seed=1, noise_std=0.02),
        rate="lo",
    )
    est = estimate_shaft_frequency(rec.vibration, rec.fs, rpm_hint=1440.0)
    assert abs(est.f_shaft_hz - 24.0) < 1.0
    assert est.confidence > 0.1


def test_full_profile_has_vibration_features():
    meta = PumpMeta(rpm=1440.0, n_vanes=6, rated_current_a=10.0)
    rec = generate_record(Condition.HEALTHY, meta=meta, config=SynthConfig(duration_s=2.0, seed=2), rate="lo")
    fv = extract_features(
        rec.vibration,
        rec.fs,
        current_rms=rec.current_rms,
        meta=FeatureMeta(
            rpm=1440.0,
            n_vanes=6,
            rated_current_a=10.0,
            bearing=BearingGeometry(8, 7.0, 35.0),
            profile="full",
        ),
    )
    assert any(n.startswith("vib_") for n in fv.names)
    assert "current_rms" in fv.names
    assert "vpf" in fv.names
    assert "vpf_minus_1x" in fv.names


def test_ct_only_strips_vibration():
    meta = PumpMeta(rpm=1440.0, rated_current_a=10.0)
    rec = generate_record(Condition.HEALTHY, meta=meta, config=SynthConfig(duration_s=1.0, seed=3), rate="lo")
    fv = extract_features(
        rec.vibration,
        rec.fs,
        current_rms=rec.current_rms,
        meta=FeatureMeta(rated_current_a=10.0, profile="ct_only"),
    )
    assert all(not n.startswith("vib_") for n in fv.names)
    assert "current_rms" in fv.names
    assert "current_rms_ratio" in fv.names


def test_no_pf_without_voltage():
    meta = PumpMeta(rated_current_a=10.0, voltage_available=False)
    rec = generate_record(Condition.HEALTHY, meta=meta, config=SynthConfig(duration_s=1.0, seed=4), rate="lo")
    fv = extract_features(
        None,
        None,
        current_rms=rec.current_rms,
        meta=FeatureMeta(rated_current_a=10.0, voltage_available=False, profile="ct_only"),
    )
    assert "power_factor_proxy" not in fv.names


def test_metadata_optional_still_produces_features():
    rec = generate_record(
        Condition.HEALTHY,
        config=SynthConfig(duration_s=2.0, seed=5),
        rate="lo",
    )
    fv = extract_features(
        rec.vibration,
        rec.fs,
        current_rms=rec.current_rms,
        meta=FeatureMeta(profile="full"),  # no rpm, no vanes, no bearing
    )
    assert len(fv.names) >= 10
    assert "vpf" not in fv.names  # needs n_vanes


def _ct_only_features(condition, severity=0.7, seed=11, onset_s=0.12, **meta_kw):
    """Extract the ct_only vector for a condition, high rate so MCSA is meaningful."""
    meta = PumpMeta(
        rpm=1440.0,
        n_vanes=6,
        rated_current_a=10.0,
        bearing=BearingGeometry(8, 7.0, 35.0),
        **meta_kw,
    )
    rec = generate_record(
        condition,
        severity=severity,
        onset_s=onset_s,
        meta=meta,
        config=SynthConfig(duration_s=0.5, seed=seed),
        rate="hi",
    )
    fv = extract_features(
        None,
        None,
        current_rms=rec.current_rms,
        current_waveform=rec.current_waveform,
        fs_current=rec.fs,
        meta=FeatureMeta(
            rpm=1440.0,
            n_vanes=6,
            rated_current_a=10.0,
            bearing=BearingGeometry(8, 7.0, 35.0),
            profile="ct_only",
        ),
    )
    return fv.as_dict()


def test_ct_only_is_not_two_scalars():
    """Regression: ct_only used to emit only current_rms and current_rms_ratio.

    DESIGN §0.3 calls ct_only the honest headline profile for submersible pumps;
    it cannot separate fault classes from a single scalar.
    """
    feats = _ct_only_features(Condition.HEALTHY)
    assert len(feats) >= 15, f"ct_only collapsed to {len(feats)} features"
    assert sum(k.startswith("mcsa_") for k in feats) >= 8


def test_mcsa_unbalance_raises_1x_sidebands():
    """Once-per-rev torque ripple must show up as f_line ± f_shaft on the current."""
    healthy = _ct_only_features(Condition.HEALTHY)
    unbal = _ct_only_features(Condition.UNBALANCE, severity=0.9)
    assert unbal["mcsa_sb_1x_sum"] > 3.0 * healthy["mcsa_sb_1x_sum"]


def test_mcsa_misalignment_raises_2x_more_than_1x():
    """Misalignment is 2x-dominant in torque, as in vibration."""
    healthy = _ct_only_features(Condition.MISALIGNMENT, severity=0.0)
    mis = _ct_only_features(Condition.MISALIGNMENT, severity=0.9)
    assert mis["mcsa_sb_2x_sum"] / (healthy["mcsa_sb_2x_sum"] + 1e-12) > 3.0


def test_mcsa_impeller_damage_raises_vpf_1x_sidebands():
    """The impeller-damage discriminator, mirrored onto the supply line."""
    healthy = _ct_only_features(Condition.HEALTHY)
    damaged = _ct_only_features(Condition.IMPELLER_DAMAGE, severity=0.9)
    assert damaged["mcsa_vpf_1x_sb_sum"] > 3.0 * healthy["mcsa_vpf_1x_sb_sum"]


def test_mcsa_cavitation_raises_noise_floor_not_lines():
    """Cavitation is stochastic: a raised broadband floor, not discrete sidebands."""
    healthy = _ct_only_features(Condition.HEALTHY)
    cav = _ct_only_features(Condition.CAVITATION, severity=0.4)
    assert cav["mcsa_noise_to_fundamental"] > healthy["mcsa_noise_to_fundamental"]


def test_current_trajectory_captures_dry_run_drop():
    """A level-only feature misses that dry-run is a transition."""
    healthy = _ct_only_features(Condition.HEALTHY)
    dry = _ct_only_features(Condition.DRY_RUN, severity=0.5)
    assert dry["current_traj_drop_ratio"] < 0.8
    assert healthy["current_traj_drop_ratio"] > 0.9


def test_mcsa_absent_without_waveform():
    """No current waveform → no invented MCSA features."""
    rec = generate_record(
        Condition.HEALTHY, config=SynthConfig(duration_s=0.5, seed=7), rate="hi"
    )
    fv = extract_features(
        None,
        None,
        current_rms=rec.current_rms,
        meta=FeatureMeta(rpm=1440.0, rated_current_a=10.0, profile="ct_only"),
    )
    assert not any(n.startswith("mcsa_") for n in fv.names)


def test_feature_matrix_alignment():
    recs = [
        generate_record(Condition.HEALTHY, config=SynthConfig(duration_s=1.0, seed=i), rate="lo")
        for i in range(3)
    ]
    vecs = [
        extract_features(r.vibration, r.fs, current_rms=r.current_rms, meta=FeatureMeta(rpm=1440, n_vanes=6, rated_current_a=10, profile="full"))
        for r in recs
    ]
    X, names = feature_matrix(vecs)
    assert X.shape == (3, len(names))


def test_iso_velocity_is_in_physical_units():
    """A pure 1 g sinusoid at f has velocity amplitude g/(2*pi*f); RMS is that/sqrt(2)."""
    from pumpwatch.features import _iso_velocity_rms_mm_s

    fs, f0, dur = 5000.0, 50.0, 2.0
    t = np.arange(int(fs * dur)) / fs
    accel_g = np.sin(2 * np.pi * f0 * t)  # 1 g amplitude
    expected_mm_s = (9.80665 / (2 * np.pi * f0)) / np.sqrt(2) * 1000.0
    got = _iso_velocity_rms_mm_s(accel_g, fs)
    assert got == pytest.approx(expected_mm_s, rel=0.05), f"{got} vs {expected_mm_s}"


def test_envelope_bandpass_rejects_shaft_content():
    """Envelope analysis without a band-pass measures shaft harmonics, not bearings."""
    from pumpwatch.features import _envelope_spectrum

    fs, dur = 26700.0, 0.5
    t = np.arange(int(fs * dur)) / fs
    f_shaft, f_defect, carrier = 24.0, 107.0, 4000.0
    # Large 1x content plus a small defect-modulated resonance.
    x = 5.0 * np.sin(2 * np.pi * f_shaft * t)
    x += 0.2 * (1 + np.sign(np.sin(2 * np.pi * f_defect * t))) * np.sin(
        2 * np.pi * carrier * t
    )
    f, spec = _envelope_spectrum(x, fs)

    def peak(target, bw=4.0):
        m = (f >= target - bw) & (f <= target + bw)
        return float(spec[m].max()) if m.any() else 0.0

    assert peak(f_defect) > peak(f_shaft), "shaft content dominates the envelope"
