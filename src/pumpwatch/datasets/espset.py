"""ESPset loader — real field data from electrical submersible pumps.

    Pellegrini, M., Varejao, F., Rodrigues, A., Mello, L.H.S. (2024). ESPset.
    Mendeley Data, V3. DOI 10.17632/m268jsw339.3 — CC BY 4.0
    Code and further detail: https://github.com/NINFA-UFES/ESPset

Why this dataset earns its place here: it is **real, in-service, multi-machine**
data from submersible centrifugal pumps. The Twente set is a controlled rig with
seeded faults on two pumps; ESPset is 11 distinct machines in offshore oil
service, which is a genuine leave-one-machine-out axis rather than a two-point
one. That makes it the only source in this project where a cross-machine claim
(C2) can be tested at a sample size where the statistics mean anything.

What it is, precisely — these constraints are load-bearing:

* **Spectra, not waveforms.** Each row is a 12103-bin spectrum. There is no time
  series, so time-domain statistics (kurtosis, crest factor), envelope/Hilbert
  bearing analysis, and any resampling are all impossible. Do not route this
  through the vibration feature extractor, which expects a waveform.
* **Order-normalised.** Every spectrum is scaled by its own running speed so bin
  3003 is always the 1x harmonic and 6006 is 2x; resolution is 1/3003 orders per
  bin, spanning to ~4.03x. This is a gift for cross-machine work — order features
  are speed-invariant by construction and no rpm metadata is needed — but it also
  means absolute frequencies are unrecoverable, so bearing defect frequencies
  (which are not integer orders and need the actual Hz axis plus geometry) cannot
  be computed.
* **Velocity in inches/second**, not acceleration in g. Mixing these units with
  the accelerometer pipeline would repeat exactly the scaling error that made
  ``iso_vel_rms`` meaningless. Converted to mm/s on load.
* **Vibration only — there is no current channel.** ESPset therefore cannot test
  the ``ct_only`` profile or anything MCSA-based, and cannot speak to dry running.
  It tests the vibration tier and cross-machine generalisation, nothing else.

Class caveat: ``Faulty sensor`` is an instrumentation-quality label, not a machine
condition. It is kept, but :func:`load_espset` exposes ``drop_sensor_faults`` and
the default taxonomy maps it to its own class rather than quietly folding a data
problem in among the mechanical faults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

ESPSET_DOI = "10.17632/m268jsw339.3"
ESPSET_CITATION = (
    "Pellegrini, M., Varejao, F., Rodrigues, A., Mello, L.H.S. (2024). ESPset. "
    "Mendeley Data, V3. DOI 10.17632/m268jsw339.3 (CC BY 4.0)"
)
ESPSET_LICENCE = "CC BY 4.0"

# Index of the 1x harmonic in every spectrum row; resolution is 1/3003 orders.
BINS_PER_ORDER = 3003
N_BINS = 12103
N_RECORDS = 6032

IN_S_TO_MM_S = 25.4

# Native labels → project taxonomy. `faulty_sensor` deliberately keeps its own
# identity: folding an instrumentation problem in with mechanical faults would
# teach a classifier to call a broken accelerometer a broken pump.
ESPSET_LABEL_MAP = {
    "Normal": "healthy",
    "Unbalance": "unbalance",
    "Misalignment": "misalignment",
    "Rubbing": "rubbing",
    "Faulty sensor": "faulty_sensor",
}

# Features published with the dataset, computed by its authors. Kept distinct from
# anything derived here so a paper can say which it used.
PUBLISHED_FEATURE_COLUMNS = [
    "median(8,13)",
    "rms(98,102)",
    "median(98,102)",
    "peak1x",
    "peak2x",
    "a",
    "b",
]


class ESPsetNotAvailableError(FileNotFoundError):
    def __init__(self, root: Path):
        super().__init__(
            f"ESPset not found at {root}.\n"
            f"Download features.zip and spectrum.zip from "
            f"https://doi.org/{ESPSET_DOI} and extract features.csv and "
            f"spectrum.csv into that directory.\n"
            f"Cite: {ESPSET_CITATION}"
        )


@dataclass
class ESPsetData:
    """Order-domain spectra plus labels and machine ids.

    `spectrum` is (n_records, n_bins) in mm/s. `orders` is the matching x-axis in
    shaft orders, so `spectrum[:, np.argmin(abs(orders - 2))]` is the 2x amplitude
    for every machine regardless of its running speed.
    """

    spectrum: np.ndarray
    orders: np.ndarray
    labels: np.ndarray
    machine_ids: np.ndarray
    record_ids: np.ndarray
    published_features: Optional[np.ndarray] = None
    published_feature_names: list[str] = field(default_factory=list)
    source: str = "espset"

    def __post_init__(self) -> None:
        n = len(self.labels)
        if not (len(self.machine_ids) == n and self.spectrum.shape[0] == n):
            raise ValueError("spectrum, labels and machine_ids must align row-wise")

    @property
    def n_machines(self) -> int:
        return len(set(self.machine_ids.tolist()))

    def describe(self) -> dict:
        classes, counts = np.unique(self.labels, return_counts=True)
        return {
            "n_records": int(len(self.labels)),
            "n_machines": self.n_machines,
            "n_bins": int(self.spectrum.shape[1]),
            "max_order": float(self.orders[-1]),
            "class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
            "units": "mm/s (velocity), order-normalised",
            "licence": ESPSET_LICENCE,
            "citation": ESPSET_CITATION,
        }


def espset_available(root: Path | str) -> bool:
    root = Path(root)
    return (root / "features.csv").exists() and (root / "spectrum.csv").exists()


def _spectrum_cache_path(root: Path) -> Path:
    return root / "spectrum_f32.npy"


def _load_spectrum(root: Path, force_reparse: bool = False) -> np.ndarray:
    """Parse the 1.8 GB CSV once, then reuse a float32 .npy (~290 MB).

    Re-parsing text on every run costs minutes and buys nothing; the values are
    single-precision at source.
    """
    cache = _spectrum_cache_path(root)
    if cache.exists() and not force_reparse:
        return np.load(cache, mmap_mode="r")

    rows = []
    with open(root / "spectrum.csv") as fh:
        for line in fh:
            line = line.strip().rstrip(";")
            if not line:
                continue
            rows.append(np.fromstring(line, sep=";", dtype=np.float64))
    spec = np.asarray(rows, dtype=np.float32)
    if spec.ndim != 2:
        raise ValueError(f"ragged spectrum rows; got shape {spec.shape}")
    np.save(cache, spec)
    return spec


def order_axis(n_bins: int = N_BINS) -> np.ndarray:
    """Shaft-order axis. Bin `BINS_PER_ORDER` is exactly 1x."""
    return np.arange(n_bins, dtype=float) / BINS_PER_ORDER


def load_espset(
    root: Path | str,
    label_map: Optional[dict] = None,
    drop_sensor_faults: bool = False,
    force_reparse: bool = False,
) -> ESPsetData:
    """Load ESPset from a directory holding features.csv and spectrum.csv."""
    root = Path(root)
    if not espset_available(root):
        raise ESPsetNotAvailableError(root)

    import csv

    with open(root / "features.csv", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        rows = list(reader)
    if len(rows) != N_RECORDS:
        # Not fatal — a future version may differ — but it must be visible.
        import warnings

        warnings.warn(
            f"expected {N_RECORDS} ESPset records, found {len(rows)}; "
            "verify the dataset version matches the cited DOI",
            stacklevel=2,
        )

    mapping = label_map or ESPSET_LABEL_MAP
    unknown = sorted({r["label"] for r in rows} - set(mapping))
    if unknown:
        raise ValueError(
            f"no taxonomy mapping for ESPset labels {unknown}; add them to "
            f"ESPSET_LABEL_MAP rather than letting raw labels into a classifier"
        )

    labels = np.array([mapping[r["label"]] for r in rows])
    machine_ids = np.array([f"esp_{int(r['esp_id']):02d}" for r in rows])
    record_ids = np.array([int(r["id"]) for r in rows])

    feats, names = None, []
    if all(c in rows[0] for c in PUBLISHED_FEATURE_COLUMNS):
        names = list(PUBLISHED_FEATURE_COLUMNS)
        feats = np.array(
            [[float(r[c]) for c in names] for r in rows], dtype=np.float64
        )

    spectrum = np.asarray(_load_spectrum(root, force_reparse), dtype=np.float32)
    if spectrum.shape[0] != len(labels):
        raise ValueError(
            f"spectrum has {spectrum.shape[0]} rows but features.csv has "
            f"{len(labels)} — the two files must be row-aligned"
        )
    spectrum = spectrum * np.float32(IN_S_TO_MM_S)  # in/s → mm/s

    data = ESPsetData(
        spectrum=spectrum,
        orders=order_axis(spectrum.shape[1]),
        labels=labels,
        machine_ids=machine_ids,
        record_ids=record_ids,
        published_features=feats,
        published_feature_names=names,
    )

    if drop_sensor_faults:
        keep = data.labels != mapping["Faulty sensor"]
        data = ESPsetData(
            spectrum=data.spectrum[keep],
            orders=data.orders,
            labels=data.labels[keep],
            machine_ids=data.machine_ids[keep],
            record_ids=data.record_ids[keep],
            published_features=None if feats is None else feats[keep],
            published_feature_names=names,
        )
    return data


# ---------------------------------------------------------------------------
# Order-domain features
# ---------------------------------------------------------------------------

# Orders that carry diagnostic meaning for a rotodynamic pump. Sub-synchronous
# content (< 1x) is where rub and instability live; 0.5x is the classic looseness
# / oil-whirl marker; integer orders carry unbalance and misalignment.
DEFAULT_ORDERS = [0.5, 1.0, 2.0, 3.0, 4.0]
DEFAULT_ORDER_BANDS = {
    "sub_synchronous": (0.05, 0.9),
    "around_1x": (0.9, 1.1),
    "around_2x": (1.9, 2.1),
    "inter_1x_2x": (1.1, 1.9),
    "above_2x": (2.1, 4.0),
}


def _band_sum(spec: np.ndarray, orders: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (orders >= lo) & (orders < hi)
    if not np.any(m):
        return np.zeros(spec.shape[0])
    return spec[:, m].sum(axis=1)


def _peak_near_order(
    spec: np.ndarray, orders: np.ndarray, target: float, half_width: float = 0.02
) -> np.ndarray:
    m = (orders >= target - half_width) & (orders <= target + half_width)
    if not np.any(m):
        return np.zeros(spec.shape[0])
    return spec[:, m].max(axis=1)


def espset_order_features(
    data: ESPsetData,
    orders_of_interest: Optional[list[float]] = None,
    bands: Optional[dict] = None,
) -> tuple[np.ndarray, list[str]]:
    """Speed-invariant order-domain features from the spectra.

    Every feature except the overall level is expressed as a *fraction of total
    spectral energy*, so it does not depend on sensor gain or mounting — which is
    what has to hold for an in-context reference set to transfer between machines.
    """
    spec = np.asarray(data.spectrum, dtype=np.float64)
    orders = data.orders
    orders_of_interest = orders_of_interest or DEFAULT_ORDERS
    bands = bands or DEFAULT_ORDER_BANDS

    total = spec.sum(axis=1) + 1e-30
    cols: dict[str, np.ndarray] = {
        "overall_level_mm_s": np.sqrt((spec**2).sum(axis=1)),
    }
    for o in orders_of_interest:
        key = f"order_{str(o).replace('.', 'p')}x"
        cols[key] = _peak_near_order(spec, orders, o) / total
    for name, (lo, hi) in bands.items():
        cols[f"band_{name}"] = _band_sum(spec, orders, lo, hi) / total

    # 1x-relative ratios: the shape a diagnostician actually reads.
    one_x = _peak_near_order(spec, orders, 1.0) + 1e-30
    cols["ratio_2x_1x"] = _peak_near_order(spec, orders, 2.0) / one_x
    cols["ratio_sub_1x"] = _band_sum(spec, orders, 0.05, 0.9) / one_x

    # Spectral shape.
    p = spec / total[:, None]
    cols["spectral_centroid_order"] = (p * orders[None, :]).sum(axis=1)
    cols["spectral_entropy"] = -(p * np.log(p + 1e-30)).sum(axis=1)

    names = sorted(cols)
    X = np.column_stack([cols[n] for n in names])
    return X, names


def write_cache_manifest(root: Path | str, data: ESPsetData) -> Path:
    """Record provenance beside the cache so a stale cache is detectable."""
    root = Path(root)
    path = root / "espset_manifest.json"
    path.write_text(json.dumps({**data.describe(), "doi": ESPSET_DOI}, indent=2))
    return path
