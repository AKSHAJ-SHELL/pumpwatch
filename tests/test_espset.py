"""Tests for the ESPset loader.

Most run against a small synthetic fixture in the real file format so the parsing
contract is enforced without the 1.8 GB download; the ones marked `real_espset`
run only when the actual dataset is present and check facts that come from the
published description.
"""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.datasets.espset import (
    BINS_PER_ORDER,
    ESPSET_LABEL_MAP,
    IN_S_TO_MM_S,
    ESPsetNotAvailableError,
    espset_available,
    espset_order_features,
    load_espset,
    order_axis,
)

real_espset = pytest.mark.skipif(
    not espset_available("data/espset"), reason="real ESPset not downloaded"
)


@pytest.fixture
def fake_espset(tmp_path):
    """Minimal dataset in ESPset's real on-disk format (semicolon CSV, trailing ;)."""
    n_bins, n_rows = 60, 12
    rng = np.random.default_rng(0)
    labels = (["Normal"] * 6) + (["Unbalance"] * 3) + (["Rubbing"] * 3)
    esp_ids = [0, 0, 1, 1, 2, 2] * 2
    cols = [
        "id", "esp_id", "label", "median(8,13)", "rms(98,102)",
        "median(98,102)", "peak1x", "peak2x", "a", "b",
    ]
    lines = [";".join(cols)]
    for i in range(n_rows):
        lines.append(
            ";".join([str(i), str(esp_ids[i]), labels[i]] + [f"{rng.random():.6f}" for _ in range(7)])
        )
    (tmp_path / "features.csv").write_text("\n".join(lines) + "\n")

    spec_lines = []
    for i in range(n_rows):
        row = rng.random(n_bins) * 0.01
        # Put a clear 1x peak at the canonical bin so order indexing is testable.
        row[BINS_PER_ORDER % n_bins] = 1.0
        spec_lines.append(";".join(f"{v:.6e}" for v in row) + ";")
    (tmp_path / "spectrum.csv").write_text("\n".join(spec_lines) + "\n")
    return tmp_path


def test_missing_dataset_raises_with_instructions(tmp_path):
    assert not espset_available(tmp_path)
    with pytest.raises(ESPsetNotAvailableError, match="10.17632"):
        load_espset(tmp_path)


def test_order_axis_puts_1x_at_the_documented_bin():
    """The dataset guarantees bin 3003 is 1x and 6006 is 2x."""
    orders = order_axis(12103)
    assert orders[BINS_PER_ORDER] == pytest.approx(1.0)
    assert orders[2 * BINS_PER_ORDER] == pytest.approx(2.0)


def test_loads_and_aligns(fake_espset):
    with pytest.warns(UserWarning, match="expected 6032"):
        d = load_espset(fake_espset)
    assert len(d.labels) == 12
    assert d.spectrum.shape[0] == 12
    assert d.n_machines == 3
    assert set(d.labels) == {"healthy", "unbalance", "rubbing"}
    assert d.published_features.shape == (12, 7)


def test_units_converted_to_mm_per_second(fake_espset):
    """Source is inches/second. Mixing units with the g-based pipeline is how
    iso_vel_rms became meaningless in the first place."""
    with pytest.warns(UserWarning):
        d = load_espset(fake_espset)
    # The injected 1x peak was 1.0 in/s.
    assert d.spectrum.max() == pytest.approx(IN_S_TO_MM_S, rel=1e-3)


def test_spectrum_cache_is_written_and_reused(fake_espset):
    with pytest.warns(UserWarning):
        load_espset(fake_espset)
    cache = fake_espset / "spectrum_f32.npy"
    assert cache.exists()
    # Corrupt the CSV; a second load must come from the cache.
    (fake_espset / "spectrum.csv").write_text("garbage\n")
    with pytest.warns(UserWarning):
        d2 = load_espset(fake_espset)
    assert d2.spectrum.shape[0] == 12


def test_unknown_label_is_refused(fake_espset):
    text = (fake_espset / "features.csv").read_text().replace("Rubbing", "Explosion")
    (fake_espset / "features.csv").write_text(text)
    with pytest.raises(ValueError, match="no taxonomy mapping"):
        load_espset(fake_espset)


def test_sensor_faults_can_be_dropped(fake_espset):
    text = (fake_espset / "features.csv").read_text().replace("Rubbing", "Faulty sensor")
    (fake_espset / "features.csv").write_text(text)
    with pytest.warns(UserWarning):
        kept = load_espset(fake_espset)
    with pytest.warns(UserWarning):
        dropped = load_espset(fake_espset, drop_sensor_faults=True)
    assert "faulty_sensor" in set(kept.labels)
    assert "faulty_sensor" not in set(dropped.labels)
    assert len(dropped.labels) == len(kept.labels) - 3


def test_sensor_fault_is_not_folded_in_with_mechanical_faults():
    """An instrumentation problem must not be taught as a pump condition."""
    assert ESPSET_LABEL_MAP["Faulty sensor"] == "faulty_sensor"
    assert ESPSET_LABEL_MAP["Faulty sensor"] not in {"healthy", "unbalance", "rubbing"}


def test_order_features_are_finite_and_named(fake_espset):
    with pytest.warns(UserWarning):
        d = load_espset(fake_espset)
    X, names = espset_order_features(d)
    assert X.shape == (12, len(names))
    assert np.isfinite(X).all()
    assert len(set(names)) == len(names)


def test_order_features_are_scale_invariant(fake_espset):
    """A change in sensor gain must not move the normalised features.

    This is the property that has to hold for a reference set to transfer between
    machines at all.
    """
    with pytest.warns(UserWarning):
        d = load_espset(fake_espset)
    X1, names = espset_order_features(d)
    d.spectrum = d.spectrum * 7.5
    X2, _ = espset_order_features(d)
    level = names.index("overall_level_mm_s")
    others = [i for i in range(len(names)) if i != level]
    assert np.allclose(X1[:, others], X2[:, others], rtol=1e-5)
    assert np.allclose(X2[:, level], X1[:, level] * 7.5, rtol=1e-5)


@real_espset
def test_real_dataset_matches_published_description():
    d = load_espset("data/espset")
    assert d.spectrum.shape == (6032, 12103)
    assert d.n_machines == 11
    assert d.orders[BINS_PER_ORDER] == pytest.approx(1.0)
    # Field prevalence: heavily healthy, which is why accuracy is not the headline.
    counts = d.describe()["class_counts"]
    assert counts["healthy"] / sum(counts.values()) > 0.75


@real_espset
def test_real_dataset_supports_eleven_fold_lomo():
    from pumpwatch.splits import split_lomo

    d = load_espset("data/espset")
    folds = split_lomo(d.machine_ids.tolist()).folds
    assert len(folds) == 11
    for f in folds:
        assert f.held_out not in set(d.machine_ids[f.context_idx].tolist())
