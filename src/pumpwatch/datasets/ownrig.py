"""Own-rig schema for collected dry-run / cavitation / impeller data.

Session metadata is mandatory for leakage-safe splits. Dry-run labels from this
rig must NEVER be merged into a cross-source classifier with Twente faults —
that creates a perfect class–machine confound.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


OWNRIG_CONDITIONS = [
    "healthy",
    "dry_run",
    "cavitation_mild",
    "cavitation_developed",
    "cavitation_choked",
    "impeller_damage",
    "bearing_wear",
]


@dataclass
class OwnRigSessionMeta:
    """Everything needed for leakage-safe splits and confound audits."""

    session_id: str
    pump_id: str
    impeller_id: str
    bearing_id: str
    mounting_type: str  # stud | adhesive | magnet
    condition: str
    severity: float
    rpm: float
    suction_valve_pct: float
    discharge_valve_pct: float
    ambient_temp_c: float
    seal_temp_c: float
    timestamp_utc: str
    voltage_available: bool = False
    n_vanes: Optional[int] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.condition not in OWNRIG_CONDITIONS:
            raise ValueError(f"unknown condition {self.condition}; expected one of {OWNRIG_CONDITIONS}")
        if self.condition == "dry_run" and self.seal_temp_c <= 0:
            raise ValueError("dry_run sessions require instrumented seal_temp_c")


@dataclass
class OwnRigRecord:
    meta: OwnRigSessionMeta
    vibration: Optional[np.ndarray] = None
    current_rms: Optional[np.ndarray] = None
    current_waveform: Optional[np.ndarray] = None
    fs: float = 26700.0
    source: str = "ownrig"


@dataclass
class SealTempCutoff:
    """Hard safety cutoff for dry-run collection."""

    max_seal_temp_c: float = 80.0
    max_exposure_s: float = 20.0

    def should_abort(self, seal_temp_c: float, exposure_s: float) -> bool:
        return seal_temp_c >= self.max_seal_temp_c or exposure_s >= self.max_exposure_s


def save_session(root: Path | str, record: OwnRigRecord) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sid = record.meta.session_id
    meta_path = root / f"{sid}.json"
    meta_path.write_text(json.dumps(asdict(record.meta), indent=2))
    if record.vibration is not None:
        np.save(root / f"{sid}_vib.npy", record.vibration)
    if record.current_rms is not None:
        np.save(root / f"{sid}_current_rms.npy", record.current_rms)
    if record.current_waveform is not None:
        np.save(root / f"{sid}_current_wave.npy", record.current_waveform)
    return meta_path


def load_ownrig(root: Path | str) -> list[OwnRigRecord]:
    root = Path(root)
    records = []
    for meta_path in sorted(root.glob("*.json")):
        meta = OwnRigSessionMeta(**json.loads(meta_path.read_text()))
        sid = meta.session_id
        vib_p = root / f"{sid}_vib.npy"
        cur_p = root / f"{sid}_current_rms.npy"
        wave_p = root / f"{sid}_current_wave.npy"
        records.append(
            OwnRigRecord(
                meta=meta,
                vibration=np.load(vib_p) if vib_p.exists() else None,
                current_rms=np.load(cur_p) if cur_p.exists() else None,
                current_waveform=np.load(wave_p) if wave_p.exists() else None,
            )
        )
    return records


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
