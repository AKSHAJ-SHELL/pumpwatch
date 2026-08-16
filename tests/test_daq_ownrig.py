"""Tests for rig acquisition, the seal interlock, and the own-rig round trip.

The interlock is the only safety-relevant control loop in this repo: DESIGN §0.2
puts seal destruction under 60 s of dry running, and the dry-run set is collected
by deliberately causing that. Its abort path therefore needs to be exercised here
rather than for the first time with a real pump running dry.
"""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.datasets.ownrig import (
    OwnRigRecord,
    OwnRigSessionMeta,
    SealTempCutoff,
    load_ownrig,
    now_utc_iso,
    save_session,
)
from pumpwatch.node.daq import SimulatedDAQ, collect_session


def _meta(condition="healthy", **kw):
    base = dict(
        session_id="s1",
        pump_id="P1",
        impeller_id="I1",
        bearing_id="B1",
        mounting_type="stud",
        condition=condition,
        severity=0.5,
        rpm=1470.0,
        suction_valve_pct=100.0,
        discharge_valve_pct=100.0,
        ambient_temp_c=25.0,
        seal_temp_c=25.0,
        timestamp_utc=now_utc_iso(),
        n_vanes=6,
    )
    base.update(kw)
    return OwnRigSessionMeta(**base)


def test_healthy_session_runs_to_completion():
    daq = SimulatedDAQ(condition="healthy", fs=4000.0)
    r = collect_session(daq, _meta("healthy"), duration_s=1.0, block_s=0.25)
    assert not r.aborted
    assert r.exposure_s == pytest.approx(1.0, rel=0.05)
    assert r.record.vibration is not None and len(r.record.vibration) > 0


def test_dry_run_aborts_on_seal_temperature():
    """The whole reason this module exists."""
    daq = SimulatedDAQ(condition="dry_run", fs=4000.0, dry_run_heating_c_per_s=20.0)
    r = collect_session(
        daq, _meta("dry_run"), duration_s=60.0, block_s=0.25,
        cutoff=SealTempCutoff(max_seal_temp_c=80.0, max_exposure_s=999.0),
    )
    assert r.aborted
    assert "seal" in r.abort_reason.lower()
    # Must stop well inside the 60 s seal budget, not at the requested duration.
    assert r.exposure_s < 10.0


def test_dry_run_aborts_on_exposure_limit_even_when_cool():
    daq = SimulatedDAQ(condition="dry_run", fs=4000.0, dry_run_heating_c_per_s=0.0)
    r = collect_session(
        daq, _meta("dry_run"), duration_s=60.0, block_s=0.25,
        cutoff=SealTempCutoff(max_seal_temp_c=500.0, max_exposure_s=2.0),
    )
    assert r.aborted
    assert r.exposure_s == pytest.approx(2.0, abs=0.3)


def test_abort_keeps_the_data_collected_so_far():
    """A destroyed seal with no recording is the worst possible outcome."""
    daq = SimulatedDAQ(condition="dry_run", fs=4000.0, dry_run_heating_c_per_s=20.0)
    r = collect_session(
        daq, _meta("dry_run"), duration_s=60.0, block_s=0.25,
        cutoff=SealTempCutoff(max_seal_temp_c=80.0, max_exposure_s=999.0),
    )
    assert r.aborted
    assert r.record.vibration is not None
    assert len(r.record.vibration) > 0
    assert r.record.current_rms is not None


def test_stop_pump_is_called_before_returning():
    called = []
    daq = SimulatedDAQ(condition="dry_run", fs=4000.0, dry_run_heating_c_per_s=20.0)
    collect_session(
        daq, _meta("dry_run"), duration_s=60.0, block_s=0.25,
        cutoff=SealTempCutoff(max_seal_temp_c=80.0, max_exposure_s=999.0),
        stop_pump=lambda reason: called.append(reason),
    )
    assert called, "actuation hook was never invoked on breach"
    assert "interlock" in called[0]


def test_already_hot_rig_never_starts():
    """A rig hot from a previous run must not begin another dry-run session."""
    called = []
    daq = SimulatedDAQ(condition="dry_run", fs=4000.0, ambient_temp_c=200.0)
    r = collect_session(
        daq, _meta("dry_run"), duration_s=5.0, block_s=0.25,
        stop_pump=lambda reason: called.append(reason),
    )
    assert r.aborted
    assert "pre-start" in r.abort_reason
    assert r.n_blocks == 0
    assert called


def test_non_dry_run_is_not_exposure_limited():
    """Only dry running is the hazard; a cavitation run should not be truncated."""
    daq = SimulatedDAQ(condition="cavitation", fs=4000.0)
    r = collect_session(
        daq, _meta("cavitation_mild"), duration_s=3.0, block_s=0.25,
        cutoff=SealTempCutoff(max_seal_temp_c=80.0, max_exposure_s=0.5),
    )
    assert not r.aborted
    assert r.exposure_s == pytest.approx(3.0, rel=0.05)


def test_recorded_seal_temp_is_measured_not_declared():
    """Metadata must carry the temperature reached, not the one typed at a prompt."""
    daq = SimulatedDAQ(condition="dry_run", fs=4000.0, dry_run_heating_c_per_s=10.0)
    meta = _meta("dry_run", seal_temp_c=25.0)
    r = collect_session(
        daq, meta, duration_s=60.0, block_s=0.25,
        cutoff=SealTempCutoff(max_seal_temp_c=70.0, max_exposure_s=999.0),
    )
    assert r.record.meta.seal_temp_c > 25.0
    assert r.record.meta.seal_temp_c == pytest.approx(r.peak_seal_temp_c)


def test_block_size_must_be_positive():
    with pytest.raises(ValueError, match="block_s"):
        collect_session(SimulatedDAQ(fs=4000.0), _meta(), duration_s=1.0, block_s=0.0)


def test_ownrig_round_trip(tmp_path):
    daq = SimulatedDAQ(condition="healthy", fs=4000.0)
    r = collect_session(daq, _meta("healthy"), duration_s=1.0, block_s=0.25)
    save_session(tmp_path, r.record)

    loaded = load_ownrig(tmp_path)
    assert len(loaded) == 1
    rec = loaded[0]
    assert rec.meta.session_id == "s1"
    assert rec.meta.condition == "healthy"
    assert np.allclose(rec.vibration, r.record.vibration)
    assert rec.current_waveform is not None


def test_dry_run_session_requires_instrumented_seal_temp():
    """Schema-level guard: a dry-run label without seal temperature is not data."""
    with pytest.raises(ValueError, match="seal_temp"):
        _meta("dry_run", seal_temp_c=0.0)
