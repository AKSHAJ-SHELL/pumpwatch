"""Tests for the CIRA loader, focused on the two silent defects in the published files.

Both would corrupt a results file without raising anything, which is why they are
pinned here rather than left to a comment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pumpwatch.datasets.cira import (
    CHANNELS,
    CIRA_CITATION,
    CiraNotAvailableError,
    CorruptedCiraFileError,
    cira_available,
    load_cira,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cira"
needs_data = pytest.mark.skipif(not cira_available(DATA), reason="CIRA data not present")


def test_absent_data_raises_with_download_instructions(tmp_path):
    assert not cira_available(tmp_path / "nothing")
    with pytest.raises(CiraNotAvailableError) as exc:
        load_cira(tmp_path / "nothing")
    assert "zenodo" in str(exc.value).lower()
    assert "10.5281/zenodo.15301820" in CIRA_CITATION


def test_european_locale_file_parses(tmp_path):
    """A_2024-10-30 is semicolon-delimited with comma decimals.

    Read with the wrong convention pandas returns one string column and no error, so a
    naive loader silently drops a pump-day. The delimiter must be detected, not assumed.
    """
    p = tmp_path / "A_2024-01-01.csv"
    cols = ";".join(f"A_{c}" for c in CHANNELS)
    p.write_text(
        f"Timestamp;{cols};Barometer;Temperature\n"
        "2024-01-01T00:00:00Z;" + ";".join(["0,5"] * len(CHANNELS)) + ";1013,2;20,1\n"
        "2024-01-01T00:00:01Z;" + ";".join(["0,6"] * len(CHANNELS)) + ";1013,3;20,2\n"
    )
    recs = load_cira(tmp_path)
    assert len(recs) == 1
    assert recs[0].values.shape == (2, len(CHANNELS))
    assert recs[0].values[0, 0] == pytest.approx(0.5)


def test_thousands_grouped_decimals_are_refused_not_coerced(tmp_path):
    """C_2024-10-30 carries values like 19.194.183.349.609.300.

    That is thousands-grouping applied to a number that already had a decimal point.
    The original decimal position is not recoverable syntactically - the same pattern
    needs two integer digits for a temperature and four for a barometer - so guessing
    would put invented numbers into a results file.
    """
    p = tmp_path / "C_2024-01-01.csv"
    cols = ",".join(f"C_{c}" for c in CHANNELS)
    good = ",".join(["0.5"] * (len(CHANNELS) - 1))
    p.write_text(
        f"Timestamp,{cols},Barometer,Temperature\n"
        f"2024-01-01T00:00:00Z,{good},19.194.183.349.609.300,1013.2,20.1\n"
        f"2024-01-01T00:00:01Z,{good},19.015.625,1013.3,20.2\n"
    )
    with pytest.raises(CorruptedCiraFileError, match="did not parse as numbers"):
        load_cira(p.parent, skip_corrupted=False)
    # Skipping every file leaves nothing readable, which must not be reported as a
    # successful load of zero records.
    with pytest.raises(CiraNotAvailableError):
        load_cira(p.parent, skip_corrupted=True)


def test_trailing_blank_line_does_not_poison_the_span(tmp_path):
    """The 2024-04-10 files end with a blank line, which becomes a NaT timestamp.

    Left in, it makes duration_hours a nonsense magnitude rather than raising.
    """
    p = tmp_path / "B_2024-01-01.csv"
    cols = ",".join(f"B_{c}" for c in CHANNELS)
    vals = ",".join(["0.5"] * len(CHANNELS))
    p.write_text(
        f"Timestamp,{cols},Barometer,Temperature\n"
        f"2024-01-01T00:00:00Z,{vals},1013.2,20.1\n"
        f"2024-01-01T01:00:00Z,{vals},1013.3,20.2\n\n"
    )
    r = load_cira(tmp_path)[0]
    assert len(r.timestamps) == 2
    assert r.duration_hours == pytest.approx(1.0)


@needs_data
def test_real_data_loads_eight_pump_days_over_three_pumps():
    recs = load_cira(DATA)
    assert len(recs) == 8, "expected 8 readable pump-days; C_2024-10-30 is corrupted"
    assert sorted({r.pump_id for r in recs}) == ["A", "B", "C"]
    for r in recs:
        assert np.all(np.diff(r.timestamps.astype("int64")) >= 0), "not time-ordered"
        assert r.duration_hours > 0
        assert r.values.shape[1] == len(CHANNELS)


@needs_data
def test_dropout_is_present_but_bounded():
    """Wireless telemetry loses packets. That is realism, not corruption.

    Pinned so that a future loader change which silently interpolates the gaps - or one
    that drops whole records because of them - shows up here.
    """
    recs = load_cira(DATA)
    fracs = [r.missing_fraction for r in recs]
    assert max(fracs) > 0, "expected some dropout; gap-free data would be suspicious"
    assert max(fracs) < 0.05, f"dropout above 5% would need explaining: {max(fracs):.3f}"


@needs_data
def test_real_data_has_the_time_axis_espset_lacks():
    """The whole reason this dataset is here: a real acquisition clock.

    Persistence rules over a trailing window are only meaningful with one.
    """
    recs = load_cira(DATA)
    for r in recs:
        dt = np.diff(r.timestamps.astype("datetime64[s]").astype("int64"))
        assert np.median(dt) == 1, "expected 1 Hz sampling"
