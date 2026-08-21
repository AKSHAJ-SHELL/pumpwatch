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

# Pump specifications, read off the Grundfos datasheets shipped in
# Appendices/Other/Datasheets. Recorded because they let the motor -> pump mapping
# above be *verified* rather than assumed: the NK80-250 is a 4-pole machine rated
# 1475 rpm and Motor-2 runs at 1480 rpm at 100%; the NK80-160 is 2-pole rated
# 2950 rpm and Motor-4's 2075 rpm is 70% of that. Both match, so MOTOR_TO_PUMP is
# evidence-backed. See test_pump_specs_confirm_the_motor_to_pump_mapping.
#
# The rated duty points are here for future NPSH work: the datasheets carry NPSH
# curves, but only as plot graphics, so NPSHr at the rated flow still has to be
# read off the curve by eye before cavitation_severity_from_npsh can be applied to
# real Twente cavitation runs.
PUMP_SPECS = {
    "NK80-250": {
        "product_name": "NK 80-250/270",
        "product_no": "98476530",
        "rated_speed_rpm": 1475.0,
        "poles": 4,
        "rated_flow_m3h": 120.3,
        "rated_head_m": 23.28,
        "impeller_diameter_mm": 270.0,      # nominal 250, trimmed to 270
        "shaft_diameter_mm": 32.0,
        "rated_power_kw": 11.0,
        "n_vanes": None,                    # not published — see below
    },
    "NK80-160": {
        "product_name": "NK 80-160/167",
        "product_no": "98663370",
        "rated_speed_rpm": 2950.0,
        "poles": 2,
        "rated_flow_m3h": 200.4,
        "rated_head_m": 28.56,
        "impeller_diameter_mm": 167.0,      # nominal 160
        "shaft_diameter_mm": 24.0,
        "rated_power_kw": 22.0,
        "n_vanes": None,
    },
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

# Impeller vane counts are unavailable for this dataset, and both routes to them
# have been tried and have failed:
#
#   1. The pump datasheets in Appendices/Other/Datasheets were extracted and their
#      text layers searched, including the full spare-parts breakdown. They give
#      impeller diameter, casting material, the impeller spare part number
#      (96591299 / 98451561) and "Number of poles" — but no vane or blade count.
#      Grundfos does not publish it: their own product literature describes a
#      "closed impeller with double-curved blades" without ever stating how many.
#   2. Estimating Z from the healthy spectra (see estimate_vane_count) is
#      inconclusive on the ch1 accelerometer: Motor-2 splits between order 4 and
#      order 6 depending on speed with no consistent 2Z harmonic, and Motor-4's
#      strongest order is 2, which is not a plausible vane count.
#
# Next thing to try: ch1 may be a motor-end sensor, and vane pass shows best on a
# pump-end one. Extract another vibration channel and re-run estimate_vane_count.
#
# Until then these stay None, so VPF, VPF-sideband and impeller-damage features
# degrade out rather than being computed at a guessed frequency.
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


def estimate_vane_count(
    records: list[TwenteRawRecord],
    z_range: tuple[int, int] = (3, 12),
    prominence_ratio: float = 2.0,
    harmonic_ratio: float = 1.5,
) -> dict:
    """Infer impeller vane count Z from healthy spectra, or report that it cannot.

    Why this exists: the 4TU pump datasheets give impeller diameter, material and
    pole count but **no vane count**, so VPF, VPF-sideband and impeller-damage
    features have no frequency to sit at. Z is in principle recoverable from the
    data — VPF = Z x f_shaft, and f_shaft is known exactly from the measurement
    overview — so this tries, and says so plainly when the evidence is weak.

    A candidate Z must satisfy two conditions, because a strong line at some
    integer order is not by itself a vane-pass line: shaft harmonics, misalignment
    (2x) and electrical content all produce integer-order peaks.

    1. the order-Z peak stands above its local spectral floor, and
    2. the order-2Z peak does too — a real vane-pass line has harmonics.

    Agreement across operating speeds is the strongest evidence available, since a
    structural resonance sits at a fixed frequency and therefore moves in *order*
    when the speed changes, whereas a true vane-pass line does not.

    Returns the evidence, not a bare integer. `confident` is False unless one Z
    wins at every speed, and the caller is expected to leave n_vanes as None in
    that case rather than guessing.
    """
    by_speed: dict[tuple, list[dict]] = {}
    for r in records:
        if r.condition != "healthy" or r.vibration is None or r.rpm is None:
            continue
        x = np.asarray(r.vibration, dtype=float)
        n = len(x)
        if n < 1024:
            continue
        spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / r.fs)
        df = float(freqs[1] - freqs[0])
        bw = max(3.0 * df, 0.6)
        f1 = r.rpm / 60.0

        prof = {}
        for z in range(z_range[0], 2 * z_range[1] + 1):
            target = z * f1
            peak = (freqs >= target - bw) & (freqs <= target + bw)
            local = (freqs >= target - 12 * bw) & (freqs <= target + 12 * bw)
            prof[z] = (
                float(spec[peak].max() / (np.median(spec[local]) + 1e-12))
                if peak.any() else 0.0
            )
        by_speed.setdefault((r.motor, r.speed_pct), []).append(prof)

    per_speed: dict[str, dict] = {}
    for key, profs in by_speed.items():
        mean = {z: float(np.mean([p[z] for p in profs])) for z in profs[0]}
        floor = float(np.median(list(mean.values())))
        cands = []
        for z in range(z_range[0], z_range[1] + 1):
            zz = mean.get(2 * z, 0.0)
            if mean[z] > prominence_ratio * floor and zz > harmonic_ratio * floor:
                cands.append({"Z": z, "order_Z": mean[z], "order_2Z": zz,
                              "score": mean[z] * zz})
        cands.sort(key=lambda c: -c["score"])
        per_speed[f"{key[0]}_{key[1]}"] = {
            "n_bursts": len(profs),
            "order_profile": {str(z): round(mean[z], 2) for z in sorted(mean)},
            "candidates": cands[:3],
        }

    by_motor: dict[str, dict] = {}
    for key, info in per_speed.items():
        motor = key.rsplit("_", 1)[0]
        tops = by_motor.setdefault(motor, {"speeds": [], "top_per_speed": []})
        tops["speeds"].append(key)
        tops["top_per_speed"].append(info["candidates"][0]["Z"] if info["candidates"] else None)

    for motor, info in by_motor.items():
        tops = [z for z in info["top_per_speed"] if z is not None]
        agreed = len(set(tops)) == 1 and len(tops) == len(info["top_per_speed"]) and tops
        info["n_vanes"] = tops[0] if agreed else None
        info["confident"] = bool(agreed and len(tops) >= 2)
        info["note"] = (
            f"Z={tops[0]} agreed across {len(tops)} speeds"
            if agreed else
            "no single Z wins at every speed — leave n_vanes as None so VPF "
            "features degrade out rather than being computed at a guessed frequency"
        )

    return {"per_speed": per_speed, "per_motor": by_motor}
