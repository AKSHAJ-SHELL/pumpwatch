"""Tests for the synthetic signal generator."""

from __future__ import annotations

import numpy as np
import pytest

from pumpwatch.physics import shaft_frequency_hz, vane_pass_frequency_hz
from pumpwatch.synth import (
    Condition,
    PumpMeta,
    SynthConfig,
    generate_dataset,
    generate_record,
)


def test_generate_healthy_record_shapes():
    rec = generate_record(Condition.HEALTHY, config=SynthConfig(duration_s=1.0, seed=1), rate="lo")
    assert rec.vibration.shape == rec.t.shape
    assert rec.current_rms.shape == rec.t.shape
    assert rec.fs == pytest.approx(1670.0)


def test_dry_run_current_mean_below_rated():
    meta = PumpMeta(rated_current_a=10.0)
    rec = generate_record(
        Condition.DRY_RUN,
        severity=1.0,
        onset_s=0.3,
        meta=meta,
        config=SynthConfig(duration_s=2.0, seed=2, noise_std=0.01),
        rate="lo",
    )
    late = rec.current_rms[rec.t > 1.5]
    assert late.mean() < 0.6 * meta.rated_current_a


def test_impeller_damage_has_sideband_energy():
    meta = PumpMeta(rpm=1440.0, n_vanes=6)
    rec = generate_record(
        Condition.IMPELLER_DAMAGE,
        severity=0.9,
        meta=meta,
        config=SynthConfig(duration_s=2.5, seed=3, noise_std=0.01),
        rate="lo",
    )
    f1 = shaft_frequency_hz(meta.rpm)
    vpf = vane_pass_frequency_hz(meta.rpm, 6)
    # Welch-ish periodogram via rFFT
    n = len(rec.vibration)
    spec = np.abs(np.fft.rfft(rec.vibration * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / rec.fs)

    def band_power(f0: float, bw: float = 1.0) -> float:
        mask = (freqs >= f0 - bw) & (freqs <= f0 + bw)
        return float(spec[mask].sum())

    sb = band_power(vpf - f1) + band_power(vpf + f1)
    healthy = generate_record(
        Condition.HEALTHY,
        meta=meta,
        config=SynthConfig(duration_s=2.5, seed=4, noise_std=0.01),
        rate="lo",
    )
    spec_h = np.abs(np.fft.rfft(healthy.vibration * np.hanning(n)))

    def band_power_h(f0: float, bw: float = 1.0) -> float:
        mask = (freqs >= f0 - bw) & (freqs <= f0 + bw)
        return float(spec_h[mask].sum())

    sb_h = band_power_h(vpf - f1) + band_power_h(vpf + f1)
    assert sb > 2.0 * sb_h


def test_cavitation_non_monotonic_in_rms():
    meta = PumpMeta()
    rms = {}
    for sev in (0.0, 0.4, 1.0):
        rec = generate_record(
            Condition.CAVITATION if sev > 0 else Condition.HEALTHY,
            severity=sev,
            meta=meta,
            config=SynthConfig(duration_s=1.0, seed=5, noise_std=0.02),
            rate="hi",
        )
        rms[sev] = float(np.std(rec.vibration))
    assert rms[0.4] > rms[0.0]
    # Choked should be below developed peak (allowing noise margin)
    assert rms[1.0] < rms[0.4] * 1.15


def test_dataset_multi_pump():
    ds = generate_dataset(n_per_class=2, n_pumps=2, seed=7)
    pumps = {r.meta.pump_id for r in ds}
    assert pumps == {"synth_0", "synth_1"}
    assert len(ds) == 2 * 2 * 7  # pumps × per_class × conditions


def test_cavitation_noise_is_band_limited():
    """Cavitation energy must land in 1-6 kHz, not across the whole spectrum.

    Full-band noise contaminates the ISO 10-1000 Hz severity band and every
    time-domain statistic, making cavitation detectable by features that have no
    physical business seeing it.
    """
    meta = PumpMeta(rpm=1440.0, n_vanes=6, rated_current_a=10.0)
    cfg = SynthConfig(duration_s=0.5, seed=3, noise_std=0.005)
    healthy = generate_record(Condition.HEALTHY, meta=meta, config=cfg, rate="hi")
    cav = generate_record(
        Condition.CAVITATION, severity=0.4, meta=meta, config=cfg, rate="hi"
    )

    def band_energy(x, fs, lo, hi):
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), d=1.0 / fs)
        return float(spec[(f >= lo) & (f < hi)].sum())

    fs = cav.fs
    in_band = band_energy(cav.vibration, fs, 1000, 6000) / (
        band_energy(healthy.vibration, fs, 1000, 6000) + 1e-30
    )
    out_band = band_energy(cav.vibration, fs, 8000, 13000) / (
        band_energy(healthy.vibration, fs, 8000, 13000) + 1e-30
    )
    assert in_band > 10.0, "cavitation energy missing from its own band"
    assert out_band < in_band / 10.0, "cavitation energy leaked far outside 1-6 kHz"


def test_dry_run_depth_varies_with_severity():
    """The 30-60% under-current band must actually be sampled, not fixed at one point."""
    meta = PumpMeta(rpm=1440.0, rated_current_a=10.0)
    depths = []
    for sev in (0.0, 0.5, 1.0):
        rec = generate_record(
            Condition.DRY_RUN,
            severity=sev,
            onset_s=0.1,
            meta=meta,
            config=SynthConfig(duration_s=3.0, seed=5),
            rate="lo",
        )
        depths.append(float(np.mean(rec.current_rms[-100:])) / meta.rated_current_a)
    assert depths[0] > depths[1] > depths[2], f"depth not monotone in severity: {depths}"
    # Residual current between 40% and 70% of rated, i.e. a 30-60% drop.
    assert 0.35 < min(depths) and max(depths) < 0.75, depths
