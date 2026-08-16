"""Tests for LoRa airtime and energy models."""

from __future__ import annotations

import pytest

from pumpwatch.node.airtime import (
    LoRaConfig,
    airtime_s,
    feature_vector_payload_bytes,
    payload_symbols,
    symbol_time_s,
)
from pumpwatch.node.energy import event_triggered_energy, fixed_schedule_energy


def test_symbol_time_sf9_bw125():
    # Tsym = 2^9 / 125e3 = 4.096 ms
    assert symbol_time_s(9, 125_000) == pytest.approx(0.004096)


def test_airtime_increases_with_payload_and_sf():
    a_small = airtime_s(20, LoRaConfig(sf=9))
    a_large = airtime_s(168, LoRaConfig(sf=9))
    a_sf12 = airtime_s(168, LoRaConfig(sf=12))
    assert a_large > a_small
    assert a_sf12 > a_large
    # 168 B @ SF9 is well above the old hand-waved 0.40 s
    assert a_large > 0.35


def test_feature_payload_bytes():
    assert feature_vector_payload_bytes(30, dtype_bytes=4, header_bytes=8) == 8 + 120


def test_payload_symbols_known_reference():
    # Sanity: empty-ish payload still has 8 symbol floor
    n = payload_symbols(1, sf=7)
    assert n >= 8


def test_fixed_schedule_15min_vs_1min():
    e15 = fixed_schedule_energy(interval_s=900.0, n_features=30)
    e1 = fixed_schedule_energy(interval_s=60.0, n_features=30)
    assert e1.mAh_per_day > 10 * e15.mAh_per_day
    assert "FAILS" in e15.notes


def test_event_triggered_primary_model():
    e = event_triggered_energy(pump_runtime_hours_per_day=3.0, n_features=30)
    assert e.model == "event_triggered"
    assert e.mAh_per_day > 0
    # At 3 h/day continuous CUSUM, should be >> the old 0.9 mAh/day fiction
    assert e.mAh_per_day > 2.0
    # But still multi-year on 2400 mAh usable at moderate runtime
    e_short = event_triggered_energy(pump_runtime_hours_per_day=2.0)
    assert e_short.battery_years > 0.5
