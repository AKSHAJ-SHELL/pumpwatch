"""CIRA centrifugal pump dataset — three industrial pumps, continuously monitored.

    Martone, A., Zazzaro, G. et al. (2025). Sensor-Based Monitoring Data from an
    Industrial System of Centrifugal Pumps. Data 10(6):91.
    Zenodo DOI 10.5281/zenodo.15301820 — CC BY 4.0

Three centrifugal pumps (A, B, C) supplying demineralised water to the boilers of a
heating plant at the Italian Aerospace Research Centre, monitored over three operational
days through a WirelessHART IoT network at one sample per second.

**This dataset carries no fault labels.** It is operational monitoring data, so it can
say nothing about recall or classification. What it has that neither ESPset nor Twente
has is a real time axis on real in-service pumps, which makes it the only data available
here that can test two things the operating-point argument currently concedes:

  1. the false-alarm rate a commissioned gate actually produces on continuously
     monitored industrial pumps, rather than on a held-out healthy split; and
  2. whether a persistence rule (k of the last n windows) suppresses false alarms as
     predicted — untestable on ESPset, whose records are independent measurements with
     no acquisition timestamps.

Both are the false-alarm half only. Nothing here bounds recall.

It also carries something the bench datasets do not: **telemetry dropout**. Between
0.01 % and 2 % of samples per pump-day are missing, because the plant reports over a
wireless mesh and packets are lost. Those gaps are preserved rather than interpolated
(see ``CiraRecord.missing_fraction``), since gap-free data is a property of a laboratory
and not of a deployment.

Two defects in the published files, both of which fail silently under a naive
``read_csv`` and are handled explicitly here.

``A_2024-10-30.csv`` is written in a European locale — semicolon delimiters, comma
decimal separators — while the other eight use commas and decimal points. Read with the
wrong convention it yields a single column of ~216k unparsed strings and pandas reports
no error. The delimiter is therefore detected per file rather than assumed.

``C_2024-10-30.csv`` is **corrupted and is refused**. Several columns carry values like
``19.194.183.349.609.300``, which is thousands-grouping applied to a number that already
had a decimal point. The original decimal position cannot be recovered syntactically:
the same pattern needs two integer digits for a temperature and four for a barometric
pressure, so any reconstruction would be a guess dressed as data. The dataset README
independently states that pump C was switched off that day, so the file should arguably
not exist. Eight pump-days remain, which is what this loader returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CIRA_DOI = "10.5281/zenodo.15301820"
CIRA_CITATION = (
    "Martone, A., Zazzaro, G. et al. (2025). Sensor-Based Monitoring Data from an "
    "Industrial System of Centrifugal Pumps. Data 10(6):91. "
    "Zenodo DOI 10.5281/zenodo.15301820 (CC BY 4.0)"
)
CIRA_LICENCE = "CC BY 4.0"

PUMPS = ("A", "B", "C")

#: Per-pump channels, with the unit prefix stripped. Barometer and ambient Temperature
#: are shared across pumps and deliberately excluded: they are environment, not machine,
#: and a gate that learns them would escalate on the weather.
CHANNELS = (
    "ACR_Mot.PV",   # motor accelerometer displacement, m/s
    "ACR_Mot.SV",   # motor accelerometer peak, m/s^2
    "ACR_Mot.TV",   # motor accelerometer contact temperature, degC
    "ACR_Pmp.PV",   # pump accelerometer displacement, m/s
    "ACR_Pmp.SV",   # pump accelerometer peak, m/s^2
    "ACR_Pmp.TV",   # pump accelerometer contact temperature, degC
    "Pres.PV",      # outlet fluid pressure, bar
    "Temp.PV",      # motor casing temperature, degC
)


class CiraNotAvailableError(FileNotFoundError):
    """Raised with download instructions rather than inventing data."""


class CorruptedCiraFileError(ValueError):
    """A published file whose numbers cannot be recovered without guessing."""


@dataclass
class CiraRecord:
    """One pump on one operational day, as a regularly sampled multichannel series."""

    pump_id: str
    day: str
    timestamps: np.ndarray      # datetime64[s], ascending
    values: np.ndarray          # (n_samples, len(CHANNELS))
    channels: tuple[str, ...] = CHANNELS

    @property
    def duration_hours(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        span = (self.timestamps[-1] - self.timestamps[0]).astype("timedelta64[s]")
        return float(span.astype(float)) / 3600.0

    @property
    def session_id(self) -> str:
        return f"{self.pump_id}_{self.day}"

    @property
    def missing_fraction(self) -> float:
        """Share of channel samples that are absent.

        Between 0.01 % and 2 % per pump-day here. This is wireless telemetry dropout,
        not corruption: the plant reports over a WirelessHART mesh and packets are
        lost. It is left in rather than interpolated, because gap-free data is exactly
        what a bench dataset has and a deployment does not, and a gate that cannot cope
        with dropout would fail in the field while scoring well on ESPset.
        """
        if self.values.size == 0:
            return 0.0
        return float((~np.isfinite(self.values)).sum()) / float(self.values.size)


def _download_hint(root: Path) -> str:
    return (
        f"CIRA pump data not found under {root}.\n"
        f"Download the nine CSVs from https://zenodo.org/records/15301820 "
        f"(CC BY 4.0, ~27 MB total) into that directory.\n"
        f"Cite: {CIRA_CITATION}"
    )


def cira_available(root: Path | str) -> bool:
    return bool(sorted(Path(root).glob("?_*.csv"))) if Path(root).exists() else False


def _read_one(path: Path, pump: str) -> "tuple[np.ndarray, np.ndarray]":
    """Parse one file, detecting its delimiter and decimal convention.

    Detected rather than assumed: A_2024-10-30.csv is published semicolon-delimited
    with comma decimals while its eight siblings use commas and decimal points, and
    reading it with the wrong convention fails silently into one string column.
    """
    import pandas as pd

    header = path.read_text(errors="ignore").split("\n", 1)[0]
    if header.count(";") > header.count(","):
        df = pd.read_csv(path, sep=";", decimal=",")
    else:
        df = pd.read_csv(path, sep=",", decimal=".")

    wanted = [f"{pump}_{c}" for c in CHANNELS]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: expected channels {missing} absent; parsed columns were "
            f"{list(df.columns)[:4]}... — delimiter detection may have failed"
        )
    # A column that survives parsing as strings has not been read, it has been
    # mis-read. Coercing it would put invented numbers into a results file.
    unparsed = [c for c in wanted if df[c].dtype == object]
    if unparsed:
        example = df[unparsed[0]].dropna().iloc[0] if df[unparsed[0]].notna().any() else "?"
        raise CorruptedCiraFileError(
            f"{path.name}: columns {unparsed} did not parse as numbers "
            f"(e.g. {example!r}). This is thousands-grouping applied to a value that "
            f"already had a decimal point; the original decimal position is not "
            f"recoverable syntactically, so the file is refused rather than guessed at."
        )
    ts = pd.to_datetime(df.iloc[:, 0], utc=True, format="mixed").dt.tz_localize(None)
    # The 2024-04-10 files carry a trailing blank line, which becomes a NaT row. Left
    # in, it makes the record's duration nonsense (a NaT end timestamp propagates into
    # every span calculation) without raising anything.
    keep = ts.notna().to_numpy()
    if not keep.all():
        ts, df = ts[keep], df[keep]
    values = df[wanted].to_numpy(dtype=float)
    return ts.to_numpy(dtype="datetime64[s]"), values


def load_cira(root: Path | str, skip_corrupted: bool = True) -> list[CiraRecord]:
    """Load every readable pump-day under ``root``.

    ``skip_corrupted`` reports and skips unrecoverable files; set it False to raise.
    """
    root = Path(root)
    if not cira_available(root):
        raise CiraNotAvailableError(_download_hint(root))

    records: list[CiraRecord] = []
    for path in sorted(root.glob("?_*.csv")):
        pump = path.stem.split("_", 1)[0]
        if pump not in PUMPS:
            continue
        day = path.stem.split("_", 1)[1]
        try:
            ts, values = _read_one(path, pump)
        except CorruptedCiraFileError as exc:
            # Skipped loudly. A silently absent pump-day is indistinguishable from one
            # that was never published, and this one *was* published — badly.
            if not skip_corrupted:
                raise
            print(f"  [skip] {exc}")
            continue
        if len(ts) == 0:
            continue
        order = np.argsort(ts)
        records.append(
            CiraRecord(pump_id=pump, day=day, timestamps=ts[order], values=values[order])
        )
    if not records:
        raise CiraNotAvailableError(_download_hint(root))
    return records
