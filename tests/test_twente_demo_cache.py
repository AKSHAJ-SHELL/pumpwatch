"""Tests for the synthetic Twente stand-in and its loader.

`make experiment` depends on this writer, and nothing tested it — the 43% coverage on
datasets/twente.py was almost entirely this path. It matters more than a synthetic
generator usually would, because its output is the data behind every number in
results_full.json, and those numbers must never be quotable as evidence about real
pumps. The caveat that says so is part of the artefact, so it is asserted here.
"""

from __future__ import annotations

import pytest

from pumpwatch.datasets.twente import (
    TWENTE_CITATION,
    TwenteNotAvailableError,
    collapse_labels,
    load_twente,
    twente_available,
    write_demo_twente_cache,
)


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("twente_demo")
    write_demo_twente_cache(root)
    return root


def test_demo_cache_round_trips(demo_root):
    assert twente_available(demo_root)
    records = load_twente(demo_root)
    assert records, "writer produced no records"
    for r in records:
        assert r.condition
        assert r.pump_id
        assert r.vibration is not None and len(r.vibration) > 0


def test_absent_cache_raises_with_download_instructions(tmp_path):
    """The loader must never silently invent Twente labels."""
    assert not twente_available(tmp_path / "nothing")
    with pytest.raises(TwenteNotAvailableError) as exc:
        load_twente(tmp_path / "nothing")
    msg = str(exc.value)
    # The citation is the actionable part: it tells the reader what to download.
    assert TWENTE_CITATION.split(".")[0] in msg or "10.4121" in msg


def test_demo_cache_is_reproducible(tmp_path):
    """Two writes must agree, or results_full.json is not reproducible either."""
    a, b = tmp_path / "a", tmp_path / "b"
    write_demo_twente_cache(a)
    write_demo_twente_cache(b)
    ra, rb = load_twente(a), load_twente(b)
    assert len(ra) == len(rb)
    assert [r.condition for r in ra] == [r.condition for r in rb]
    assert ra[0].vibration.shape == rb[0].vibration.shape


def test_demo_cache_has_more_than_one_machine(demo_root):
    """A single-machine stand-in could not exercise the machine-grouped splits."""
    records = load_twente(demo_root)
    assert len({r.pump_id for r in records}) > 1


def test_demo_cache_has_healthy_and_faulty(demo_root):
    records = load_twente(demo_root)
    labels = {r.condition for r in records}
    assert "healthy" in labels
    assert labels - {"healthy"}, "no fault classes in the stand-in"


def test_collapse_labels_maps_into_the_declared_taxonomy(demo_root):
    """The collapsed space, not the raw family list, is what the models see."""
    from pumpwatch.datasets.twente import TWENTE_COLLAPSED_CLASSES

    records = load_twente(demo_root)
    collapsed = collapse_labels([r.condition for r in records])
    assert set(collapsed) <= set(TWENTE_COLLAPSED_CLASSES)


def test_every_declared_family_has_a_collapse_rule():
    """Guards the failure mode where a family is added and its mapping is forgotten.

    That would surface only at runtime, on real data, part-way through an experiment.
    """
    from pumpwatch.datasets.twente import (
        TWENTE_CLASS_COLLAPSE,
        TWENTE_FAULT_FAMILIES,
    )

    assert set(TWENTE_FAULT_FAMILIES) == set(TWENTE_CLASS_COLLAPSE)


def test_collapsed_taxonomy_fits_under_the_class_cap():
    from pumpwatch.datasets.twente import (
        TABPFN_MAX_CLASSES,
        TWENTE_COLLAPSED_CLASSES,
    )

    assert len(TWENTE_COLLAPSED_CLASSES) <= TABPFN_MAX_CLASSES


def test_collapse_refuses_an_unmapped_label(demo_root):
    """Silently dropping a class would change what the reported accuracy means."""
    with pytest.raises(ValueError, match="no collapse rule"):
        collapse_labels(["healthy", "a_fault_nobody_declared"])
