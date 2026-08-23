"""Paderborn University bearing dataset — an independent test for the normalisation result.

    Lessmeier, C., Kimotho, J.K., Zimmer, D., Sextro, W. (2016). Condition monitoring
    of bearing damage in electromechanical drive systems by using motor current
    signals of electric motors: a benchmark data set for data-driven classification.
    European Conference of the PHM Society.
    Zenodo mirror DOI 10.5281/zenodo.15845309 — CC BY 4.0

Why this dataset is here and not in the main evaluation: the headline normalisation
result rests on a single dataset, and ESPset is the only public source with a genuine
cross-machine axis for pumps. Paderborn is not pumps and its held-out unit is a bearing
rather than a machine, so it cannot replicate the pump claim. What it can do is test
whether the *effect* — that standardising each unit by its own statistics loses badly to
pooling the training units — appears on independently collected data, from a different
laboratory, on a different machine type, with a different sensor suite.

One property makes it an unusually sharp test. Each file here is a single physical
bearing carrying a single damage state, so per-unit normalisation centres a unit whose
records are **100 % one class**. That is the extreme of the class-skew condition that
was hypothesised, and refuted, as the mechanism on ESPset.

This loader deliberately uses the accelerated-lifetime ("real") damage bearings and
excludes the artificially damaged ones. Mixing the two confounds damage type with
everything else, and the realistic set is the one the leakage literature argues matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PADERBORN_DOI = "10.5281/zenodo.15845309"
PADERBORN_CITATION = (
    "Lessmeier, C., Kimotho, J.K., Zimmer, D., Sextro, W. (2016). Condition "
    "monitoring of bearing damage in electromechanical drive systems by using motor "
    "current signals of electric motors: a benchmark data set for data-driven "
    "classification. European Conference of the PHM Society. "
    "Zenodo DOI 10.5281/zenodo.15845309 (CC BY 4.0)"
)
PADERBORN_LICENCE = "CC BY 4.0"

FS_HIGH = 64_000.0  # vibration and phase currents

# Bearing code -> damage class. Only accelerated-lifetime damage is listed; the
# artificially damaged bearings (KA01, KA03, KA05-09, KI01, KI03, KI05, KI07, KI08)
# are deliberately absent so that damage type cannot confound the comparison.
BEARING_CLASS: dict[str, str] = {
    "K001": "healthy", "K002": "healthy", "K003": "healthy",
    "K004": "healthy", "K005": "healthy", "K006": "healthy",
    "KA04": "outer_race", "KA15": "outer_race", "KA16": "outer_race",
    "KA22": "outer_race", "KA30": "outer_race",
    "KI04": "inner_race", "KI14": "inner_race", "KI16": "inner_race",
    "KI17": "inner_race", "KI18": "inner_race", "KI21": "inner_race",
}

# Operating condition code -> (rpm, load torque Nm, radial force N).
OPERATING_CONDITIONS: dict[str, tuple[float, float, float]] = {
    "N15_M07_F10": (1500.0, 0.7, 1000.0),
    "N09_M07_F10": (900.0, 0.7, 1000.0),
    "N15_M01_F10": (1500.0, 0.1, 1000.0),
    "N15_M07_F04": (1500.0, 0.7, 400.0),
}


class PaderbornNotAvailableError(FileNotFoundError):
    """Raised with download instructions rather than inventing data."""


@dataclass
class PaderbornRecord:
    bearing_id: str          # the held-out unit — one physical bearing
    condition: str           # healthy / outer_race / inner_race
    operating_point: str     # e.g. N15_M07_F10
    rpm: float
    vibration: np.ndarray
    current: Optional[np.ndarray]
    fs: float = FS_HIGH
    measurement: int = 0

    @property
    def session_id(self) -> str:
        """One acquisition file. Record-wise splits must not span this."""
        return f"{self.operating_point}_{self.bearing_id}_{self.measurement}"


def _download_hint(root: Path) -> str:
    return (
        f"Paderborn bearing data not found under {root}.\n"
        f"Download from https://zenodo.org/records/15845309 (CC BY 4.0, ~166 MB per\n"
        f"bearing) and extract each RAR so that files land as\n"
        f"  {root}/<BEARING>/<BEARING>/<COND>_<BEARING>_<n>.mat\n"
        f"Only the accelerated-lifetime damage bearings are used: "
        f"{sorted(BEARING_CLASS)}\n"
        f"Cite: {PADERBORN_CITATION}"
    )


def paderborn_available(root: Path | str) -> bool:
    root = Path(root)
    return root.exists() and any(root.glob("*/*/*.mat"))


def load_paderborn(
    root: Path | str,
    max_per_condition: int = 5,
    window_s: float = 1.0,
    with_current: bool = True,
) -> list[PaderbornRecord]:
    """Load records, taking at most ``max_per_condition`` measurements per file group.

    Each .mat holds four seconds at 64 kHz. We take a single window of ``window_s``
    from each, rather than chopping one measurement into many, because chopping is
    exactly the segment-level leakage the evaluation protocol exists to prevent.
    """
    root = Path(root)
    if not paderborn_available(root):
        raise PaderbornNotAvailableError(_download_hint(root))

    from scipy.io import loadmat

    n = int(window_s * FS_HIGH)
    records: list[PaderbornRecord] = []
    for bearing in sorted(BEARING_CLASS):
        for cond in sorted(OPERATING_CONDITIONS):
            paths = sorted(root.glob(f"{bearing}/{bearing}/{cond}_{bearing}_*.mat"))
            for p in paths[:max_per_condition]:
                try:
                    m = loadmat(str(p), squeeze_me=True, struct_as_record=False)
                except Exception:
                    continue
                keys = [k for k in m if not k.startswith("__")]
                if not keys:
                    continue
                entries = {str(e.Name): np.asarray(e.Data).ravel() for e in m[keys[0]].Y}
                vib = entries.get("vibration_1")
                if vib is None or vib.size < n:
                    continue
                cur = entries.get("phase_current_1") if with_current else None
                if cur is not None and cur.size < n:
                    cur = None
                try:
                    meas = int(p.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    meas = 0
                records.append(
                    PaderbornRecord(
                        bearing_id=bearing,
                        condition=BEARING_CLASS[bearing],
                        operating_point=cond,
                        rpm=OPERATING_CONDITIONS[cond][0],
                        vibration=vib[:n].astype(float),
                        current=None if cur is None else cur[:n].astype(float),
                        measurement=meas,
                    )
                )
    if not records:
        raise PaderbornNotAvailableError(_download_hint(root))
    return records
