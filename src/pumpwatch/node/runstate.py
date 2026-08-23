"""Is the machine actually doing work? The question the gate never asked.

A condition monitor has nothing to say about a stopped pump, and this codebase had no
way to express that. ``node/energy.py`` assumed ``pump_runtime_hours_per_day=3.0`` and
``node/gates.py`` had no notion of the machine being off, so "CUSUM continuous *while
running*" was a sentence in a design document that nothing enforced.

The omission cost us a published finding. Commissioning the gate on a day of CIRA
telemetry that was 89-92 % idle produced a baseline describing a stopped pump; run
forward against a running one it escalated 100 % of windows, and that was written up as
evidence that plant demand defeats a commissioned baseline. It was evidence that we had
learned the wrong baseline.

**Why it never showed on the curated datasets.** ESPset and Twente records are
acquisitions somebody chose to take on a running machine. They are implicitly
run-state-gated by whoever collected them, so a monitor with no concept of "off" scores
perfectly well on both. Deployment telemetry is not curated that way. A duty-cycled pump
spends most of its day stopped, and a gate installed on one learns idle as normal the
first time it commissions.

The detector is deliberately dull: a threshold on a load-bearing channel with hysteresis
and a dwell. Motor current RMS is the natural channel on the node; outlet pressure works
on plant telemetry. What matters is not the sophistication but that the state exists and
that ``UNKNOWN`` is representable, so a caller with no load channel is told rather than
silently handed ``RUNNING``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class RunState(str, Enum):
    """Whether the machine is doing work.

    ``UNKNOWN`` is a real answer, not a placeholder. A detector with no load channel to
    look at must say so: defaulting to ``RUNNING`` is how an idle baseline gets learned,
    and defaulting to ``OFF`` is how a monitor goes silent on a working pump.
    """

    OFF = "off"
    RUNNING = "running"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunStateConfig:
    """Thresholds for the run detector.

    Two thresholds rather than one. A pump idling near a single threshold would
    otherwise chatter between states every window, and each transition resets the
    dwell counter, so the detector would never settle long enough to commission
    anything. ``on_threshold`` must exceed ``off_threshold``.
    """

    on_threshold: float
    off_threshold: float
    dwell_windows: int = 2

    def __post_init__(self) -> None:
        if self.on_threshold <= self.off_threshold:
            raise ValueError(
                f"on_threshold ({self.on_threshold}) must exceed off_threshold "
                f"({self.off_threshold}); equal thresholds give no hysteresis and the "
                f"detector will chatter on a machine idling at the boundary"
            )
        if self.dwell_windows < 1:
            raise ValueError("dwell_windows must be at least 1")


def _otsu_threshold(x: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Threshold maximising between-class variance, plus a separability score.

    A duty-cycled machine's load is bimodal — stopped or working, with little between —
    so splitting it is a two-class problem, not a quantile problem. Otsu's criterion
    finds the split without needing to be told where the modes are.

    The score returned is the between-class variance as a fraction of the total. Near 1
    the two states are cleanly separated; near 0 the load is unimodal and there is no
    meaningful run/off distinction to draw. Callers need that, because a threshold
    computed on a unimodal load is arbitrary and will slice a healthy distribution in
    half.
    """
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(float) / max(hist.sum(), 1)
    cum_w = np.cumsum(w)
    cum_m = np.cumsum(w * centres)
    total_m = cum_m[-1]
    denom = cum_w * (1.0 - cum_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (total_m * cum_w - cum_m) ** 2 / denom
    between = np.where(np.isfinite(between), between, -np.inf)
    # Take the middle of the tied region, not the first index. On a sharply bimodal
    # load every empty bin between the two modes achieves the same between-class
    # variance, so argmax returns the *left edge of the gap* — a threshold sitting on
    # the idle mode's shoulder. The midpoint is the robust choice and is what a human
    # reading the histogram would pick.
    best = between.max()
    tied = np.flatnonzero(between >= best - 1e-12 * max(abs(best), 1.0))
    k = int(tied[len(tied) // 2])
    total_var = float(x.var()) if x.size else 0.0
    score = float(between[k] / total_var) if total_var > 0 else 0.0
    return float(centres[k]), min(max(score, 0.0), 1.0)


def config_from_healthy_load(
    load: np.ndarray,
    band_fraction: float = 0.25,
    dwell_windows: int = 2,
    min_separability: float = 0.5,
) -> RunStateConfig:
    """Derive run/off thresholds from an observed load channel.

    Deriving them beats hardcoding, because the channel's units differ per deployment:
    motor current in amps, outlet pressure in bar.

    Uses Otsu rather than a quantile. An earlier version took the median of everything
    above the 5th percentile, which on these plant pumps landed at 0.70 bar against an
    idle floor of 0.5 and a running pressure of 40 — that is, *inside the idle mode*.
    It classified stopped windows as running and the resulting baseline was as wrong as
    the one it was meant to fix. On a bimodal load a quantile has no reason to fall
    between the modes; Otsu does exactly that by construction.

    ``min_separability`` refuses a load that is not convincingly bimodal, rather than
    returning an arbitrary split of a unimodal distribution.
    """
    load = np.asarray(load, dtype=float)
    finite = load[np.isfinite(load)]
    if finite.size == 0:
        raise ValueError("no finite load samples to derive thresholds from")
    if np.ptp(finite) <= 0:
        raise ValueError(
            "load channel is constant — this machine appears never to run, so no run "
            "threshold can be derived from it"
        )

    thresh, separability = _otsu_threshold(finite)
    if separability < min_separability:
        raise ValueError(
            f"load channel is not convincingly bimodal (separability {separability:.2f} "
            f"< {min_separability}); a run/off threshold on a unimodal load would slice "
            f"a healthy distribution in half. Supply a RunStateConfig explicitly if the "
            f"machine really is always running."
        )

    lo, hi = float(finite.min()), float(finite.max())
    band = band_fraction * min(thresh - lo, hi - thresh)
    return RunStateConfig(
        on_threshold=thresh + band,
        off_threshold=thresh - band,
        dwell_windows=dwell_windows,
    )


class RunStateDetector:
    """Sequential run/off detector with hysteresis and a dwell requirement."""

    def __init__(self, config: RunStateConfig, initial: RunState = RunState.OFF) -> None:
        self.config = config
        self.state = initial
        self._pending: Optional[RunState] = None
        self._pending_count = 0

    def reset(self) -> None:
        self.state = RunState.OFF
        self._pending = None
        self._pending_count = 0

    def update(self, load: Optional[float]) -> RunState:
        """Advance one window. ``None`` or a non-finite load yields ``UNKNOWN``.

        An unknown reading does **not** change the committed state. Telemetry drops
        packets, and treating a gap as a state transition would make the detector track
        the radio rather than the pump.
        """
        if load is None or not np.isfinite(load):
            return RunState.UNKNOWN

        if load >= self.config.on_threshold:
            candidate = RunState.RUNNING
        elif load <= self.config.off_threshold:
            candidate = RunState.OFF
        else:
            # Between the thresholds: hold. This is what hysteresis buys.
            self._pending, self._pending_count = None, 0
            return self.state

        if candidate == self.state:
            self._pending, self._pending_count = None, 0
            return self.state

        if candidate == self._pending:
            self._pending_count += 1
        else:
            self._pending, self._pending_count = candidate, 1

        if self._pending_count >= self.config.dwell_windows:
            self.state = candidate
            self._pending, self._pending_count = None, 0
        return self.state

    def label(self, load: np.ndarray) -> np.ndarray:
        """Run the detector over a load series, returning one ``RunState`` per window."""
        self.reset()
        return np.array([self.update(v) for v in np.asarray(load, dtype=float)], dtype=object)


def running_mask(load: np.ndarray, config: Optional[RunStateConfig] = None) -> np.ndarray:
    """Boolean mask of windows where the machine is running.

    Convenience for the common case of labelling a whole series offline. Windows that
    come back ``UNKNOWN`` are excluded: a monitor should not claim a machine was running
    on the strength of a dropped packet.
    """
    load = np.asarray(load, dtype=float)
    config = config or config_from_healthy_load(load)
    states = RunStateDetector(config).label(load)
    return np.array([s == RunState.RUNNING for s in states], dtype=bool)
