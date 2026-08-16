"""Parser for the Twente/4TU distribution as actually published.

    Kumar, D. et al. (2023). Motor current and vibration monitoring dataset for
    various faults in an E-motor-driven centrifugal pump. Data in Brief 51:109779.
    4TU.ResearchData, DOI 10.4121/2b61183e-c14f-4131-829b-cc4822c369d0 — CC BY 4.0

The existing loader in :mod:`pumpwatch.datasets.twente` reads a hand-authored
``manifest.json``. This module reads the real download, whose layout is::

    Dataset.7z
      Vibration/Motor-{2,4}/{50,70,75,100}/<condition>/Vibration_Motor-N_S_time-<cond>-chK.csv
      Electric /Motor-{2,4}/{50,70,75,100}/<condition>/Electric_Motor-N_S_time-<cond>-chK.csv

Each CSV is **one channel** and holds a ``time`` column plus many numbered
columns; each numbered column is a **separate acquisition burst**, not another
channel. Vibration files carry ~68 bursts of 12 s, electric files ~36 bursts of
15 s, both at 20 kHz. That structure is a gift for the leakage ladder: bursts
within one condition folder share a mounting and a setup, which is exactly the
session-level nuisance that separates a record-wise split from a random one.

Two structural facts about this dataset are important enough to be enforced in
code rather than left to a reader's notes:

**1. Leave-one-machine-out is impossible here.** The two motors carry almost
disjoint condition sets — Motor-2 has the bearing, impeller and electrical faults,
Motor-4 has the cavitation, alignment, unbalance and coupling faults, and the only
labels they share are the healthy variants. Holding out either motor leaves a
training set with none of the test motor's fault classes. Anything claiming a
cross-machine result on Twente alone is mistaken; see :func:`lomo_feasible`.
What Twente *does* support, and ESPset does not, is a genuine
cross-operating-condition split: Motor-2 was run at 50%, 75% and 100% speed.

**2. Vibration and current bursts are not simultaneous.** They were acquired on
different schedules, and the files even hold different burst counts. Pairing burst
*i* of a vibration file with burst *i* of a current file — which is what
``pair_channels=True`` does — is an approximation that is defensible only because
each folder is a steady-state run of one condition. It is off by default, and any
result that relies on it has to say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

TWENTE_RAW_DOI = "10.4121/2b61183e-c14f-4131-829b-cc4822c369d0"
TWENTE_RAW_MD5 = "e8507480782a8b948e4c8d78bf9d1ab2"
DEFAULT_FS = 20_000.0

# Motor -> pump. The 4TU appendices name the pumps; both are Grundfos NK80 frames.
MOTOR_TO_PUMP = {
    "Motor-2": "NK80-250",
    "Motor-4": "NK80-160",
}

# Shaft speed per (motor, speed setting), read from the dataset's own
# "measurement overview.xlsx" rather than assumed from a nameplate. Order features
# are computed at these frequencies, so a guess here would place every one of them
# at the wrong place — the same defect that made the demo cache's hardcoded
# geometry meaningless.
SPEED_RPM = {
    ("Motor-2", 50): 740.0,
    ("Motor-2", 75): 1110.0,
    ("Motor-2", 100): 1480.0,
    ("Motor-4", 70): 2075.0,
}

# Impeller vane counts are NOT recorded here. The pump datasheets ship as PDFs in
# the appendices and the vane count has not been read off them, so vane-pass and
# VPF-sideband features are unavailable on this dataset and the extractor is left
# to degrade to its no-n_vanes subset. Inventing a number would silently compute
# every VPF feature at the wrong frequency.
MOTOR_TO_N_VANES: dict[str, Optional[int]] = {
    "Motor-2": None,
    "Motor-4": None,
}

# Folder name (severity suffix stripped) -> project taxonomy.
CONDITION_MAP = {
    "healthy": "healthy",
    "healthy noise": "healthy",
    # A replacement motor fitted as a fresh baseline. Mapped to healthy, but kept
    # separately named here so the choice is visible and easy to override.
    "new motor": "healthy",
    "bearing bpfo": "bearing_outer",
    "bearing bpfi": "bearing_inner",
    "bearing bsf": "bearing_ball",
    "bearing contaminated": "bearing_contamination",
    "bearing pump": "bearing_pump",
    "impeller": "impeller_damage",
    "cavitation suction": "cavitation",
    "cavitation discharge": "cavitation",
    "align angular": "misalignment",
    "align parallel": "misalignment",
    "align combination": "misalignment",
    "unbalance motor": "unbalance",
    "unbalance pump": "unbalance",
    "loose foot motor": "loose_foot",
    "loose foot pump": "loose_foot",
    "soft foot": "soft_foot",
    "bent shaft": "bent_shaft",
    "coupling": "coupling",
    "broken rotor bar": "broken_rotor_bar",
    "stator short": "stator_short",
}

_TRAILING_INDEX = re.compile(r"\s+(\d+[A-Za-z]?)$")


def parse_condition(folder: str) -> tuple[str, str, Optional[int]]:
    """Split a condition folder into (taxonomy label, family, severity index).

    ``"bearing bpfo 3"`` -> ``("bearing_outer", "bearing bpfo", 3)``. The trailing
    index is the dataset's severity/repeat marker and is what distinguishes one
    physical faulted component from another, so it drives component-wise splits.
    """
    name = folder.strip().lower()
    m = _TRAILING_INDEX.search(name)
    severity: Optional[int] = None
    family = name
    if m:
        family = name[: m.start()].strip()
        try:
            severity = int(re.sub(r"[A-Za-z]", "", m.group(1)))
        except ValueError:
            severity = None
    if family not in CONDITION_MAP:
        raise KeyError(
            f"no taxonomy mapping for Twente condition {folder!r} (family {family!r}); "
            f"add it to CONDITION_MAP rather than guessing a label"
        )
    return CONDITION_MAP[family], family, severity


@dataclass
class TwenteRawRecord:
    """One acquisition burst with its grouping keys."""

    pump_id: str
    motor: str
    speed_pct: int
    condition: str
    family: str
    severity: Optional[int]
    burst: int
    fs: float
    vibration: Optional[np.ndarray] = None
    current: Optional[np.ndarray] = None
    source: str = "twente"

    @property
    def session_id(self) -> str:
        """One recording session = one condition folder at one speed."""
        return f"{self.motor}_{self.speed_pct}_{self.family}_{self.severity}"

    @property
    def component_id(self) -> str:
        """The physical faulted part, independent of the speed it was run at."""
        return f"{self.motor}_{self.family}_{self.severity}"

    @property
    def operating_point(self) -> str:
        return f"{self.motor}_{self.speed_pct}"

    @property
    def rpm(self) -> Optional[float]:
        """Measured shaft speed, or None if this combination is untabulated."""
        return SPEED_RPM.get((self.motor, self.speed_pct))

    @property
    def n_vanes(self) -> Optional[int]:
        return MOTOR_TO_N_VANES.get(self.motor)


def _iter_condition_dirs(root: Path, modality: str) -> Iterator[tuple[Path, str, int, str]]:
    base = root / modality
    if not base.exists():
        return
    for motor_dir in sorted(base.iterdir()):
        if not motor_dir.is_dir():
            continue
        for speed_dir in sorted(motor_dir.iterdir()):
            if not speed_dir.is_dir():
                continue
            try:
                speed = int(speed_dir.name)
            except ValueError:
                continue
            for cond_dir in sorted(speed_dir.iterdir()):
                if cond_dir.is_dir():
                    yield cond_dir, motor_dir.name, speed, cond_dir.name


def _read_burst_csv(
    path: Path,
    max_bursts: Optional[int],
    max_samples: Optional[int],
) -> tuple[np.ndarray, float]:
    """Return (bursts, fs) where bursts is (n_bursts, n_samples)."""
    import pandas as pd

    usecols = None
    if max_bursts is not None:
        # 'time' plus the first max_bursts numbered columns.
        usecols = ["time"] + [str(i) for i in range(max_bursts)]

    df = pd.read_csv(
        path,
        usecols=lambda c: usecols is None or c in usecols,
        nrows=max_samples,
        engine="c",
    )
    if "time" not in df.columns:
        raise ValueError(f"{path} has no 'time' column")
    t = df["time"].to_numpy(dtype=float)
    fs = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else DEFAULT_FS
    data = df.drop(columns=["time"]).to_numpy(dtype=np.float32).T  # (bursts, samples)
    return data, fs


def load_twente_raw(
    root: Path | str,
    channel: str = "ch1",
    max_bursts: Optional[int] = 8,
    window_s: Optional[float] = 2.0,
    pair_channels: bool = False,
    conditions: Optional[set[str]] = None,
) -> list[TwenteRawRecord]:
    """Load bursts from an extracted Twente tree.

    Parameters
    ----------
    max_bursts, window_s
        Bound the work. A single vibration CSV is ~330 MB of text holding 68
        bursts of 12 s; loading every burst of every file is rarely what you want
        and will not fit in memory.
    pair_channels
        Attach current bursts to vibration bursts by index. See the module
        docstring — the two modalities were not sampled simultaneously, so this is
        an explicitly-flagged approximation, off by default.
    """
    root = Path(root)
    max_samples = int(window_s * DEFAULT_FS) if window_s else None

    vib_by_key: dict[tuple, tuple[np.ndarray, float]] = {}
    for cond_dir, motor, speed, cond in _iter_condition_dirs(root, "Vibration"):
        if conditions and cond not in conditions:
            continue
        files = sorted(cond_dir.glob(f"*-{channel}.csv"))
        if not files:
            continue
        vib_by_key[(motor, speed, cond)] = _read_burst_csv(files[0], max_bursts, max_samples)

    cur_by_key: dict[tuple, tuple[np.ndarray, float]] = {}
    for cond_dir, motor, speed, cond in _iter_condition_dirs(root, "Electric"):
        if conditions and cond not in conditions:
            continue
        files = sorted(cond_dir.glob(f"*-{channel}.csv"))
        if not files:
            continue
        cur_by_key[(motor, speed, cond)] = _read_burst_csv(files[0], max_bursts, max_samples)

    records: list[TwenteRawRecord] = []
    for key in sorted(set(vib_by_key) | set(cur_by_key)):
        motor, speed, cond = key
        label, family, severity = parse_condition(cond)
        vib, fs_v = vib_by_key.get(key, (None, DEFAULT_FS))
        cur, fs_c = cur_by_key.get(key, (None, DEFAULT_FS))

        n = 0
        if vib is not None:
            n = max(n, vib.shape[0])
        if cur is not None:
            n = max(n, cur.shape[0])
        if pair_channels and vib is not None and cur is not None:
            n = min(vib.shape[0], cur.shape[0])

        for b in range(n):
            v = vib[b] if vib is not None and b < vib.shape[0] else None
            c = cur[b] if (pair_channels and cur is not None and b < cur.shape[0]) else None
            records.append(
                TwenteRawRecord(
                    pump_id=MOTOR_TO_PUMP.get(motor, motor),
                    motor=motor,
                    speed_pct=speed,
                    condition=label,
                    family=family,
                    severity=severity,
                    burst=b,
                    fs=fs_v if v is not None else fs_c,
                    vibration=v,
                    current=c,
                )
            )
    return records


def lomo_feasible(records: list[TwenteRawRecord]) -> dict:
    """Report whether leave-one-machine-out is possible on the loaded records.

    Answers the question the design assumed rather than checked: Twente's two
    motors carry almost disjoint fault sets, so LOMO would train on one label set
    and test on another. Returns the evidence rather than a bare boolean.
    """
    by_machine: dict[str, set[str]] = {}
    for r in records:
        by_machine.setdefault(r.pump_id, set()).add(r.condition)
    machines = sorted(by_machine)
    shared = set.intersection(*by_machine.values()) if by_machine else set()
    non_healthy_shared = sorted(shared - {"healthy"})
    return {
        "machines": machines,
        "classes_per_machine": {m: sorted(by_machine[m]) for m in machines},
        "shared_classes": sorted(shared),
        "shared_fault_classes": non_healthy_shared,
        "lomo_feasible": len(machines) >= 2 and len(non_healthy_shared) >= 1,
        "note": (
            "LOMO needs at least one FAULT class present on more than one machine; "
            "otherwise every fold trains and tests on disjoint label sets and the "
            "resulting score measures nothing."
        ),
    }
