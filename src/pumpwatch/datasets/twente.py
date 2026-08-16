"""Twente / 4TU centrifugal pump dataset loader.

Kumar et al. (2023), Data in Brief 51:109779.
DOI 10.4121/2b61183e-c14f-4131-829b-cc4822c369d0 — CC BY 4.0

This loader accepts a local cache directory. If data is absent, it raises with
download instructions — we do not silently invent Twente labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


TWENTE_DOI = "10.4121/2b61183e-c14f-4131-829b-cc4822c369d0"
TWENTE_CITATION = (
    "Kumar, D. et al. (2023). Motor current and vibration monitoring dataset "
    "for various faults in an E-motor-driven centrifugal pump. "
    "Data in Brief 51:109779. DOI 10.4121/2b61183e-c14f-4131-829b-cc4822c369d0"
)

# Fault families present in Twente (dry-run is NOT among them)
TWENTE_FAULT_FAMILIES = [
    "healthy",
    "bearing_outer",
    "bearing_inner",
    "bearing_ball",
    "bearing_contamination",
    "impeller_damage",
    "cavitation",
    "misalignment",
    "unbalance",
    "loose_foot",
    "soft_foot",
    "bent_shaft",
    "coupling",
    "broken_rotor_bar",
    "stator_short",
]


@dataclass
class TwenteRecord:
    pump_id: str
    condition: str
    severity: str
    rpm: float
    vibration: Optional[np.ndarray] = None
    current: Optional[np.ndarray] = None
    fs_vib: float = 20_000.0
    fs_current: float = 20_000.0
    session_id: str = ""
    source: str = "twente"
    meta: dict = field(default_factory=dict)


class TwenteNotAvailableError(FileNotFoundError):
    """Raised when the local Twente cache is missing."""

    def __init__(self, root: Path):
        msg = (
            f"Twente dataset not found at {root}.\n"
            f"Download from https://doi.org/{TWENTE_DOI}\n"
            f"Cite: {TWENTE_CITATION}\n"
            "Place extracted files under the given root and retry."
        )
        super().__init__(msg)


def twente_available(root: Path | str) -> bool:
    root = Path(root)
    return root.exists() and (
        (root / "manifest.json").exists()
        or any(root.glob("**/*.mat"))
        or any(root.glob("**/*.csv"))
        or any(root.glob("**/*.h5"))
    )


def load_twente_manifest(root: Path | str) -> list[dict]:
    """Load a manifest.json listing records, or raise if data absent."""
    root = Path(root)
    if not twente_available(root):
        raise TwenteNotAvailableError(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    # Fallback: scan for .npy windows with sidecar JSON metadata
    records = []
    for meta_path in sorted(root.glob("**/record_*.json")):
        records.append(json.loads(meta_path.read_text()))
    if not records:
        raise TwenteNotAvailableError(root)
    return records


def load_twente_record(root: Path | str, entry: dict) -> TwenteRecord:
    root = Path(root)
    vib = None
    cur = None
    if "vibration_path" in entry:
        vib = np.load(root / entry["vibration_path"])
    if "current_path" in entry:
        cur = np.load(root / entry["current_path"])
    return TwenteRecord(
        pump_id=entry["pump_id"],
        condition=entry["condition"],
        severity=entry.get("severity", "unknown"),
        rpm=float(entry.get("rpm", 1470.0)),
        vibration=vib,
        current=cur,
        fs_vib=float(entry.get("fs_vib", 20_000.0)),
        fs_current=float(entry.get("fs_current", 20_000.0)),
        session_id=entry.get("session_id", ""),
        meta=entry,
    )


def load_twente(root: Path | str) -> list[TwenteRecord]:
    manifest = load_twente_manifest(root)
    return [load_twente_record(root, e) for e in manifest]


def write_demo_twente_cache(root: Path | str, n_per_class: int = 5, seed: int = 0) -> Path:
    """Write a *synthetic stand-in* that matches the Twente schema for CI.

    Clearly marked source='twente_demo' — NOT the real dataset.
    Dry-run is deliberately excluded (Twente has no dry-run class).
    """
    from pumpwatch.synth import Condition, PumpMeta, SynthConfig, generate_record

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        Condition.HEALTHY: "healthy",
        Condition.CAVITATION: "cavitation",
        Condition.IMPELLER_DAMAGE: "impeller_damage",
        Condition.BEARING_OUTER: "bearing_outer",
        Condition.BEARING_INNER: "bearing_inner",
        Condition.UNBALANCE: "unbalance",
        Condition.MISALIGNMENT: "misalignment",
        Condition.LOOSENESS: "loose_foot",
    }
    rng = np.random.default_rng(seed)
    manifest = []
    idx = 0
    for pump_i, (pump_id, rpm) in enumerate([("NK80-250", 1470.0), ("NK80-160", 2950.0)]):
        meta = PumpMeta(pump_id=pump_id, rpm=rpm, rated_current_a=10.0 + pump_i)
        for cond, label in mapping.items():
            for j in range(n_per_class):
                rec = generate_record(
                    cond,
                    severity=float(rng.uniform(0.3, 0.8)) if cond != Condition.HEALTHY else 0.0,
                    meta=meta,
                    config=SynthConfig(duration_s=1.0, seed=int(rng.integers(0, 1e9)), noise_std=0.03),
                    rate="lo",
                )
                vib_path = f"vib_{idx:04d}.npy"
                cur_path = f"cur_{idx:04d}.npy"
                np.save(root / vib_path, rec.vibration)
                np.save(root / cur_path, rec.current_rms)
                entry = {
                    "pump_id": pump_id,
                    "condition": label,
                    "severity": "demo",
                    "rpm": rpm,
                    "vibration_path": vib_path,
                    "current_path": cur_path,
                    "fs_vib": rec.fs,
                    "fs_current": rec.fs,
                    "session_id": f"{pump_id}_{label}_{j}",
                    "source": "twente_demo",
                }
                manifest.append(entry)
                idx += 1
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "README.txt").write_text(
        "DEMO CACHE — synthetic stand-in for CI. NOT the real Twente dataset.\n"
        f"Real data: https://doi.org/{TWENTE_DOI}\n"
    )
    return root
