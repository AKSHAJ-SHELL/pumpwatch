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
    current_waveform: Optional[np.ndarray] = None
    fs_vib: float = 20_000.0
    fs_current: float = 20_000.0
    session_id: str = ""
    # Grouping keys for the leakage ladder. Absent keys degrade a rung to unusable
    # rather than silently falling back to a weaker split.
    component_id: str = ""
    operating_point: str = ""
    source: str = "twente"
    # Per-pump geometry. Required for VPF and bearing-envelope features to sit at
    # the right frequencies — inventing them silently computes every order-domain
    # feature at the wrong place, which is worse than not computing them.
    n_vanes: Optional[int] = None
    rated_current_a: Optional[float] = None
    bearing_n_balls: Optional[int] = None
    bearing_ball_d_mm: Optional[float] = None
    bearing_pitch_d_mm: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def bearing_geometry(self):
        """BearingGeometry if the manifest declared it, else None."""
        from pumpwatch.physics import BearingGeometry

        if None in (self.bearing_n_balls, self.bearing_ball_d_mm, self.bearing_pitch_d_mm):
            return None
        return BearingGeometry(
            self.bearing_n_balls, self.bearing_ball_d_mm, self.bearing_pitch_d_mm
        )


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

    def _load(key: str):
        return np.load(root / entry[key]) if key in entry else None

    return TwenteRecord(
        pump_id=entry["pump_id"],
        condition=entry["condition"],
        severity=entry.get("severity", "unknown"),
        rpm=float(entry.get("rpm", 1470.0)),
        vibration=_load("vibration_path"),
        current=_load("current_path"),
        current_waveform=_load("current_waveform_path"),
        fs_vib=float(entry.get("fs_vib", 20_000.0)),
        fs_current=float(entry.get("fs_current", 20_000.0)),
        session_id=entry.get("session_id", ""),
        component_id=entry.get("component_id", ""),
        operating_point=entry.get("operating_point", ""),
        source=entry.get("source", "twente"),
        n_vanes=entry.get("n_vanes"),
        rated_current_a=entry.get("rated_current_a"),
        bearing_n_balls=entry.get("bearing_n_balls"),
        bearing_ball_d_mm=entry.get("bearing_ball_d_mm"),
        bearing_pitch_d_mm=entry.get("bearing_pitch_d_mm"),
        meta=entry,
    )


def load_twente(root: Path | str) -> list[TwenteRecord]:
    manifest = load_twente_manifest(root)
    return [load_twente_record(root, e) for e in manifest]



# Two pumps with genuinely different geometry — a LOMO fold that differs only in
# rpm is a much weaker cross-machine test than one where the impeller vane count
# and the bearings differ too.
DEMO_PUMPS = [
    {"pump_id": "NK80-250", "rpm": 1470.0, "n_vanes": 6, "rated_current_a": 10.0,
     "bearing": (8, 7.0, 35.0)},
    {"pump_id": "NK80-160", "rpm": 2950.0, "n_vanes": 5, "rated_current_a": 11.0,
     "bearing": (9, 7.9, 38.5)},
]

# VFD speed settings. Without more than one operating point per pump the
# cross-operating-condition rung of the ladder cannot be evaluated at all — and for
# a VFD-driven pump that is the rung that matters most in the field.
DEMO_SPEED_RATIOS = {"50Hz": 1.0, "40Hz": 0.8}


def write_demo_twente_cache(
    root: Path | str,
    n_sessions: int = 2,
    n_windows_per_session: int = 4,
    healthy_session_multiplier: int = 6,
    seed: int = 0,
    rate: str = "hi",
    duration_s: float = 0.5,
    n_per_class: Optional[int] = None,
) -> Path:
    """Write a *synthetic stand-in* that matches the Twente schema for CI.

    Clearly marked ``source='twente_demo'`` — NOT the real dataset. Dry-run is
    deliberately excluded (Twente has no dry-run class).

    Generated at the high rate (26.7 kSPS) by default. The previous default of
    ``rate="lo"`` (1.67 kSPS, Nyquist 835 Hz) aliased the 4 kHz bearing-fault
    carrier down to ~660 Hz and left the 2-4 kHz and 4-6 kHz cavitation band
    features identically zero, so every bearing and cavitation number computed on
    this cache was measuring an artefact.

    Structure mirrors how pump data is actually collected, because the leakage
    ladder is only meaningful if the nuisance structure it exploits is present:

        pump → operating point → condition → session (a recording) → windows

    Each *session* fixes a sensor gain, a mounting resonance and a noise floor,
    and every window cut from that session inherits them. That is what a
    random-window split gets to memorise and a record-wise split does not.
    """
    from pumpwatch.physics import BearingGeometry
    from pumpwatch.synth import Condition, PumpMeta, SynthConfig, generate_record

    if n_per_class is not None:
        # Back-compat: callers that asked for N records per class get N windows
        # spread over two sessions, so the session structure still exists.
        n_sessions = 2
        n_windows_per_session = max(1, n_per_class // 2)

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
    manifest: list[dict] = []
    idx = 0
    # Two physical instances of each faulted part, so leave-one-component-out holds
    # out a real component rather than a random subset of one component's windows.
    n_components = 2

    for pump in DEMO_PUMPS:
        bearing = BearingGeometry(*pump["bearing"])
        for op_name, ratio in DEMO_SPEED_RATIOS.items():
            rpm = pump["rpm"] * ratio
            meta = PumpMeta(
                pump_id=pump["pump_id"],
                rpm=rpm,
                n_vanes=pump["n_vanes"],
                rated_current_a=pump["rated_current_a"],
                bearing=bearing,
            )
            for cond, label in mapping.items():
                # Healthy operation is what a pump does almost all of the time, and
                # the MCU gate needs n > 10p healthy windows before it can be armed
                # (see baseline_lifecycle.commissioning_length). Generating as few
                # healthy windows as faulty ones would make commissioning look
                # impossible for a reason that is an artefact of the sampling.
                n_sess = n_sessions * (healthy_session_multiplier if cond == Condition.HEALTHY else 1)
                for j in range(n_sess):
                    comp = j % n_components
                    # Session-level nuisance shared by every window from this
                    # recording: sensor gain, a mounting resonance and the noise
                    # floor all stay fixed while the cable stays plugged in.
                    sess_gain = float(np.exp(rng.normal(0.0, 0.18)))
                    sess_res_hz = float(rng.uniform(200.0, 900.0))
                    sess_res_amp = float(rng.uniform(0.05, 0.30))
                    sess_noise = float(rng.uniform(0.02, 0.06))
                    severity = (
                        float(rng.uniform(0.3, 0.8)) if cond != Condition.HEALTHY else 0.0
                    )
                    for _w in range(n_windows_per_session):
                        rec = generate_record(
                            cond,
                            severity=severity,
                            meta=meta,
                            config=SynthConfig(
                                duration_s=duration_s,
                                seed=int(rng.integers(0, 1e9)),
                                noise_std=sess_noise,
                            ),
                            rate=rate,
                        )
                        tt = np.arange(len(rec.vibration)) / rec.fs
                        vib = sess_gain * (
                            rec.vibration
                            + sess_res_amp * np.sin(2.0 * np.pi * sess_res_hz * tt)
                        )
                        cur = sess_gain * rec.current_rms
                        wave = sess_gain * rec.current_waveform

                        vib_path = f"vib_{idx:04d}.npy"
                        cur_path = f"cur_{idx:04d}.npy"
                        wave_path = f"curwave_{idx:04d}.npy"
                        # float32 keeps the high-rate cache to a sane size on disk.
                        np.save(root / vib_path, vib.astype(np.float32))
                        np.save(root / cur_path, cur.astype(np.float32))
                        np.save(root / wave_path, wave.astype(np.float32))

                        manifest.append({
                            "pump_id": pump["pump_id"],
                            "condition": label,
                            "severity": "demo",
                            "rpm": rpm,
                            "vibration_path": vib_path,
                            "current_path": cur_path,
                            "current_waveform_path": wave_path,
                            "fs_vib": rec.fs,
                            "fs_current": rec.fs,
                            # Grouping keys for the leakage ladder (levels 1-4).
                            "session_id": f"{pump['pump_id']}_{op_name}_{label}_s{j}",
                            "component_id": f"{pump['pump_id']}_{label}_c{comp}",
                            "operating_point": f"{pump['pump_id']}_{op_name}",
                            "source": "twente_demo",
                            "n_vanes": pump["n_vanes"],
                            "rated_current_a": pump["rated_current_a"],
                            "bearing_n_balls": bearing.n_balls,
                            "bearing_ball_d_mm": bearing.ball_diameter_mm,
                            "bearing_pitch_d_mm": bearing.pitch_diameter_mm,
                        })
                        idx += 1

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "README.txt").write_text(
        "DEMO CACHE — synthetic stand-in for CI. NOT the real Twente dataset.\n"
        f"Real data: https://doi.org/{TWENTE_DOI}\n"
        "Structure: pump -> operating point -> condition -> session -> windows.\n"
        "Windows within a session share a sensor gain, mounting resonance and noise\n"
        "floor, so the leakage ladder has real nuisance structure to expose.\n"
    )
    return root
