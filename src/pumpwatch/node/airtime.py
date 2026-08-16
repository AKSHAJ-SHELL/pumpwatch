"""Exact LoRa airtime (Semtech SX126x formula).

No more hand-waved "100 B @ SF9 = 0.40 s". Payload size comes from the
serialized feature vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class LoRaConfig:
    sf: int = 9  # spreading factor 7–12
    bw_hz: int = 125_000
    cr: int = 1  # coding rate denominator offset: 1→4/5 … 4→4/8
    preamble_symbols: int = 8
    explicit_header: bool = True
    low_data_rate_opt: bool = False
    crc_on: bool = True


def symbol_time_s(sf: int, bw_hz: int) -> float:
    return (2**sf) / bw_hz


def payload_symbols(
    payload_bytes: int,
    sf: int,
    cr: int = 1,
    explicit_header: bool = True,
    low_data_rate_opt: bool = False,
    crc_on: bool = True,
) -> int:
    """Number of payload symbols (Semtech AN1200.13)."""
    if not (7 <= sf <= 12):
        raise ValueError(f"SF must be 7–12, got {sf}")
    if not (1 <= cr <= 4):
        raise ValueError(f"CR must be 1–4, got {cr}")
    pl = payload_bytes
    ih = 0 if explicit_header else 1
    de = 1 if low_data_rate_opt else 0
    crc = 1 if crc_on else 0
    num = 8 * pl - 4 * sf + 28 + 16 * crc - 20 * ih
    den = 4 * (sf - 2 * de)
    n_payload = 8 + max(ceil(num / den) * (cr + 4), 0)
    return int(n_payload)


def airtime_s(payload_bytes: int, cfg: LoRaConfig | None = None) -> float:
    """Total time on air for one uplink packet."""
    cfg = cfg or LoRaConfig()
    t_sym = symbol_time_s(cfg.sf, cfg.bw_hz)
    t_preamble = (cfg.preamble_symbols + 4.25) * t_sym
    n_payload = payload_symbols(
        payload_bytes,
        cfg.sf,
        cr=cfg.cr,
        explicit_header=cfg.explicit_header,
        low_data_rate_opt=cfg.low_data_rate_opt or cfg.sf >= 11,
        crc_on=cfg.crc_on,
    )
    t_payload = n_payload * t_sym
    return t_preamble + t_payload


def feature_vector_payload_bytes(
    n_features: int,
    dtype_bytes: int = 4,
    header_bytes: int = 8,
) -> int:
    """Serialized feature vector size: header + n_features × dtype."""
    return header_bytes + n_features * dtype_bytes


def heartbeat_payload_bytes(header_bytes: int = 8) -> int:
    """Heartbeat: header + D² float."""
    return header_bytes + 4
