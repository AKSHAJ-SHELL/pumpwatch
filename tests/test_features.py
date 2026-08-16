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
