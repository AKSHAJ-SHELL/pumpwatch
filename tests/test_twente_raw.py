"""Tests for the real Twente/4TU parser and split-feasibility checking."""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.datasets.twente_raw import (
    CONDITION_MAP,
    SPEED_RPM,
    TwenteRawRecord,
    load_twente_raw,
    lomo_feasible,
    parse_condition,
)
from pumpwatch.splits import split_by_group, split_lomo, split_label_coverage, SplitLevel


@pytest.mark.parametrize(
    "folder,label,family,severity",
    [
        ("healthy 1", "healthy", "healthy", 1),
        ("bearing bpfo 3", "bearing_outer", "bearing bpfo", 3),
        ("bearing bpfi 2", "bearing_inner", "bearing bpfi", 2),
        ("bearing bsf", "bearing_ball", "bearing bsf", None),
        ("impeller 2", "impeller_damage", "impeller", 2),
        ("cavitation suction 4", "cavitation", "cavitation suction", 4),
        ("align angular 5", "misalignment", "align angular", 5),
        ("unbalance pump 1", "unbalance", "unbalance pump", 1),
        ("loose foot pump", "loose_foot", "loose foot pump", None),
        ("broken rotor bar", "broken_rotor_bar", "broken rotor bar", None),
        ("coupling 2D", "coupling", "coupling", 2),
    ],
)
def test_condition_parsing(folder, label, family, severity):
    assert parse_condition(folder) == (label, family, severity)


def test_unmapped_condition_is_refused():
    """Guessing a label for an unrecognised folder would corrupt the taxonomy."""
    with pytest.raises(KeyError, match="no taxonomy mapping"):
        parse_condition("mystery fault 1")


def test_every_mapped_family_has_a_taxonomy_label():
    assert all(isinstance(v, str) and v for v in CONDITION_MAP.values())


def test_speeds_come_from_the_measurement_overview():
    """rpm is read from the dataset's own overview, never assumed from a nameplate."""
    assert SPEED_RPM[("Motor-2", 100)] == 1480.0
    assert SPEED_RPM[("Motor-2", 50)] == 740.0
    # Scaling is consistent: 50% really is half of 100%.
    assert SPEED_RPM[("Motor-2", 50)] / SPEED_RPM[("Motor-2", 100)] == pytest.approx(0.5)


def _rec(motor, speed, family, severity, condition, burst=0):
    return TwenteRawRecord(
        pump_id={"Motor-2": "NK80-250", "Motor-4": "NK80-160"}[motor],
        motor=motor, speed_pct=speed, condition=condition, family=family,
        severity=severity, burst=burst, fs=20000.0,
        vibration=np.zeros(8), current=np.zeros(8),
    )


def test_grouping_keys_separate_session_component_and_operating_point():
    a = _rec("Motor-2", 100, "bearing bpfo", 1, "bearing_outer")
    b = _rec("Motor-2", 50, "bearing bpfo", 1, "bearing_outer")
    # Same physical component, different speed: component matches, session does not.
    assert a.component_id == b.component_id
    assert a.session_id != b.session_id
    assert a.operating_point != b.operating_point


def test_lomo_is_reported_infeasible_when_motors_share_no_fault():
    """The central structural fact about Twente.

    Motor-2 carries the bearing/impeller faults and Motor-4 the hydraulic ones;
    only the healthy variants appear on both. LOMO would train and test on
    disjoint label sets, so it measures nothing.
    """
    recs = [
        _rec("Motor-2", 100, "healthy", 1, "healthy"),
        _rec("Motor-2", 100, "bearing bpfo", 1, "bearing_outer"),
        _rec("Motor-4", 70, "healthy", 1, "healthy"),
        _rec("Motor-4", 70, "cavitation suction", 1, "cavitation"),
    ]
    out = lomo_feasible(recs)
    assert out["lomo_feasible"] is False
    assert out["shared_classes"] == ["healthy"]
    assert out["shared_fault_classes"] == []


def test_lomo_is_feasible_when_a_fault_is_shared():
    recs = [
        _rec("Motor-2", 100, "healthy", 1, "healthy"),
        _rec("Motor-2", 100, "bearing bpfo", 1, "bearing_outer"),
        _rec("Motor-4", 70, "healthy", 1, "healthy"),
        _rec("Motor-4", 70, "bearing bpfo", 1, "bearing_outer"),
    ]
    out = lomo_feasible(recs)
    assert out["lomo_feasible"] is True
    assert "bearing_outer" in out["shared_fault_classes"]


def test_missing_tree_returns_nothing(tmp_path):
    assert load_twente_raw(tmp_path) == []


# --- split feasibility -----------------------------------------------------


def test_label_coverage_flags_disjoint_folds():
    """A fold tested on a class it never trained on scores zero by construction."""
    groups = ["A"] * 4 + ["B"] * 4
    labels = ["healthy", "healthy", "bearing", "bearing"] + ["healthy"] * 2 + ["cavitation"] * 2
    result = split_by_group(groups, SplitLevel.COMPONENT_WISE, max_folds=0)
    cov = split_label_coverage(result, labels)
    assert cov["total_unseen_test_classes"] > 0
    assert not cov["interpretable"]
    unseen = {c for f in cov["folds"] for c in f["missing_from_train"]}
    assert unseen == {"bearing", "cavitation"}


def test_label_coverage_passes_a_well_posed_split():
    groups = ["A"] * 4 + ["B"] * 4
    labels = ["healthy", "bearing"] * 4
    result = split_by_group(groups, SplitLevel.COMPONENT_WISE, max_folds=0)
    cov = split_label_coverage(result, labels)
    assert cov["total_unseen_test_classes"] == 0
    assert cov["interpretable"]


def test_label_coverage_catches_the_twente_lomo_case():
    machines = ["NK80-250"] * 4 + ["NK80-160"] * 4
    labels = ["healthy", "healthy", "bearing_outer", "bearing_outer"] + [
        "healthy", "healthy", "cavitation", "cavitation",
    ]
    cov = split_label_coverage(split_lomo(machines), labels)
    assert not cov["interpretable"]
    assert cov["fraction_test_classes_unseen"] >= 0.5


# --- vane-count estimation -------------------------------------------------


def _healthy_burst(motor, speed, f_shaft, z, fs=20000.0, dur=2.0, seed=0, amp_2z=1.0):
    """Synthetic healthy burst with a vane-pass line at Z (and optionally 2Z)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    x = (0.5 * np.sin(2 * np.pi * f_shaft * t)
         + 1.0 * np.sin(2 * np.pi * z * f_shaft * t)
         + amp_2z * 0.4 * np.sin(2 * np.pi * 2 * z * f_shaft * t)
         + 0.05 * rng.standard_normal(len(t)))
    return TwenteRawRecord(
        pump_id="P", motor=motor, speed_pct=speed, condition="healthy",
        family="healthy", severity=1, burst=0, fs=fs, vibration=x,
    )


def test_vane_count_recovered_when_the_line_is_really_there(monkeypatch):
    """A clean VPF line plus its 2Z harmonic, consistent across speeds."""
    import pumpwatch.datasets.twente_raw as tr

    monkeypatch.setitem(tr.SPEED_RPM, ("MotorX", 50), 600.0)
    monkeypatch.setitem(tr.SPEED_RPM, ("MotorX", 100), 1200.0)
    recs = [_healthy_burst("MotorX", 50, 10.0, 6, seed=1),
            _healthy_burst("MotorX", 100, 20.0, 6, seed=2)]
    out = tr.estimate_vane_count(recs)
    info = out["per_motor"]["MotorX"]
    assert info["n_vanes"] == 6
    assert info["confident"] is True


def test_vane_count_refuses_when_speeds_disagree(monkeypatch):
    """Disagreement across speeds means the line is not vane pass.

    A structural resonance sits at a fixed frequency, so it moves in ORDER when
    the speed changes — which is exactly what disagreement looks like.
    """
    import pumpwatch.datasets.twente_raw as tr

    monkeypatch.setitem(tr.SPEED_RPM, ("MotorY", 50), 600.0)
    monkeypatch.setitem(tr.SPEED_RPM, ("MotorY", 100), 1200.0)
    recs = [_healthy_burst("MotorY", 50, 10.0, 6, seed=3),
            _healthy_burst("MotorY", 100, 20.0, 4, seed=4)]
    info = tr.estimate_vane_count(recs)["per_motor"]["MotorY"]
    assert info["n_vanes"] is None
    assert info["confident"] is False
    assert "degrade out" in info["note"]


def test_vane_count_requires_the_2z_harmonic(monkeypatch):
    """A lone integer-order peak is not enough — shaft and electrical content
    produce those too."""
    import pumpwatch.datasets.twente_raw as tr

    monkeypatch.setitem(tr.SPEED_RPM, ("MotorZ", 50), 600.0)
    recs = [_healthy_burst("MotorZ", 50, 10.0, 6, seed=5, amp_2z=0.0)]
    per = tr.estimate_vane_count(recs)["per_speed"]["MotorZ_50"]
    assert all(c["Z"] != 6 for c in per["candidates"])


def test_vane_count_ignores_faulty_records(monkeypatch):
    import pumpwatch.datasets.twente_raw as tr

    monkeypatch.setitem(tr.SPEED_RPM, ("MotorW", 50), 600.0)
    r = _healthy_burst("MotorW", 50, 10.0, 6, seed=6)
    r.condition = "bearing_outer"
    assert tr.estimate_vane_count([r])["per_motor"] == {}


def test_real_twente_vane_count_is_honestly_inconclusive():
    """Regression on a real finding: neither datasheet nor spectra give Z.

    If a future channel or a wider extraction makes this pass confidently, that is
    a genuine improvement and this test should be updated to assert the value.
    """
    from pumpwatch.datasets.twente_raw import MOTOR_TO_N_VANES

    assert MOTOR_TO_N_VANES == {"Motor-2": None, "Motor-4": None}
