"""DAQ abstraction for rig collection, with an interlocked acquisition loop.

The safety argument is the reason this module exists rather than a bare script.
DESIGN §0.2 puts mechanical seal destruction under 60 s of dry running, and the
whole point of the dry-run set is to record that event. A pre-flight check of seal
temperature — which is what the collection script used to do, taking one
temperature as a command-line argument and validating it before writing metadata —
cannot protect anything: the temperature that matters is the one thirty seconds
into an acquisition that has already started.

So acquisition here is a loop over short blocks, and between every block the
interlock is re-evaluated against live seal temperature and elapsed exposure. On
breach it stops the pump through the supplied callback *first*, then keeps the
partial data with the abort reason attached. A destroyed seal with no recording is
the worst outcome; a short recording with an honest abort reason is a good one.

Backends are pluggable so a real card can be dropped in without touching the
control flow. `SimulatedDAQ` exists so the entire collection path — including the
abort path, which is the part you least want to debug for the first time while a
real pump is running dry — is exercisable and testable today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from pumpwatch.datasets.ownrig import (
    OwnRigRecord,
    OwnRigSessionMeta,
    SealTempCutoff,
)


class DAQBackend(Protocol):
    """Minimum surface a data-acquisition device must present.

    Deliberately small: one block of synchronous channels plus the interlock
    channel. Anything richer is a driver detail that does not belong in the
    control flow.
    """

    fs: float

    def read_block(self, n_samples: int) -> dict[str, np.ndarray]:
        """Return a dict with at least 'vibration' and/or 'current' arrays."""
        ...

    def read_seal_temp_c(self) -> float:
        """Live seal-face temperature. This is the interlock channel."""
        ...

    def close(self) -> None: ...


@dataclass
class AcquisitionResult:
    record: OwnRigRecord
    aborted: bool
    abort_reason: str = ""
    exposure_s: float = 0.0
    peak_seal_temp_c: float = 0.0
    seal_temp_trace: list[float] = field(default_factory=list)
    n_blocks: int = 0


@dataclass
class SimulatedDAQ:
    """Physics-backed stand-in so the collection path is runnable without hardware.

    Seal temperature ramps only while the condition is a dry run, at a rate chosen
    to cross an 80 °C cutoff in tens of seconds — which is the regime the interlock
    exists for, and therefore the regime its tests need to reach.
    """

    condition: str = "healthy"
    severity: float = 0.5
    rpm: float = 1470.0
    n_vanes: int = 6
    rated_current_a: float = 10.0
    fs: float = 26_700.0
    ambient_temp_c: float = 25.0
    dry_run_heating_c_per_s: float = 6.0
    seed: int = 0

    _elapsed_s: float = 0.0
    _rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def read_block(self, n_samples: int) -> dict[str, np.ndarray]:
        from pumpwatch.synth import Condition, PumpMeta, SynthConfig, generate_record

        cond = Condition(self.condition) if self.condition in {
            c.value for c in Condition
        } else Condition.HEALTHY
        duration = n_samples / self.fs
        rec = generate_record(
            cond,
            severity=self.severity,
            onset_s=0.0 if self.condition == "dry_run" else 1e9,
            meta=PumpMeta(
                rpm=self.rpm, n_vanes=self.n_vanes, rated_current_a=self.rated_current_a
            ),
            config=SynthConfig(
                duration_s=duration, seed=int(self._rng.integers(0, 1e9)), fs_hi=self.fs
            ),
            rate="hi",
        )
        self._elapsed_s += duration
        return {
            "vibration": rec.vibration[:n_samples],
            "current_rms": rec.current_rms[:n_samples],
            "current_waveform": rec.current_waveform[:n_samples],
        }

    def read_seal_temp_c(self) -> float:
        if self.condition != "dry_run":
            return self.ambient_temp_c + 2.0
        return self.ambient_temp_c + self.dry_run_heating_c_per_s * self._elapsed_s

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


def collect_session(
    backend: DAQBackend,
    meta: OwnRigSessionMeta,
    duration_s: float,
    cutoff: Optional[SealTempCutoff] = None,
    block_s: float = 0.25,
    stop_pump: Optional[Callable[[str], None]] = None,
    sleep: Callable[[float], None] = lambda s: None,
) -> AcquisitionResult:
    """Acquire for `duration_s`, re-checking the seal interlock between blocks.

    `stop_pump` is the actuation hook — a contactor, a VFD stop, or a human-facing
    alarm. It is called BEFORE the data is finalised, because stopping the pump is
    more urgent than saving the file.

    `block_s` bounds how long the rig can run past a breach. At the default 0.25 s
    the interlock has ~240 opportunities to fire inside a 60 s seal budget; making
    it much larger trades seal life for marginally fewer temperature reads.
    """
    cutoff = cutoff or SealTempCutoff()
    if block_s <= 0:
        raise ValueError("block_s must be positive")
    n_block = max(int(round(block_s * backend.fs)), 1)

    chunks: dict[str, list[np.ndarray]] = {}
    temps: list[float] = []
    exposure = 0.0
    aborted, reason = False, ""
    n_blocks = 0
    is_dry_run = meta.condition == "dry_run"

    # Check before the first block too: the rig may already be hot from a prior run.
    t0 = backend.read_seal_temp_c()
    temps.append(t0)
    if is_dry_run and cutoff.should_abort(t0, 0.0):
        reason = f"pre-start interlock: seal {t0:.1f}C already at/above cutoff"
        if stop_pump:
            stop_pump(reason)
        return AcquisitionResult(
            record=OwnRigRecord(meta=meta),
            aborted=True,
            abort_reason=reason,
            exposure_s=0.0,
            peak_seal_temp_c=t0,
            seal_temp_trace=temps,
            n_blocks=0,
        )

    while exposure < duration_s:
        block = backend.read_block(n_block)
        for k, v in block.items():
            chunks.setdefault(k, []).append(np.asarray(v))
        exposure += n_block / backend.fs
        n_blocks += 1

        temp = backend.read_seal_temp_c()
        temps.append(temp)

        # The interlock applies to dry-run sessions, where the fault IS the hazard.
        # Other conditions still record temperature but are not exposure-limited.
        if is_dry_run and cutoff.should_abort(temp, exposure):
            aborted = True
            reason = (
                f"interlock at {exposure:.2f}s: seal {temp:.1f}C "
                f"(max {cutoff.max_seal_temp_c}C), exposure limit "
                f"{cutoff.max_exposure_s}s"
            )
            if stop_pump:
                stop_pump(reason)
            break
        sleep(block_s)

    def _cat(key: str) -> Optional[np.ndarray]:
        return np.concatenate(chunks[key]) if chunks.get(key) else None

    peak = max(temps) if temps else 0.0
    # Record the temperature actually reached, not the one typed at the prompt.
    meta.seal_temp_c = peak
    record = OwnRigRecord(
        meta=meta,
        vibration=_cat("vibration"),
        current_rms=_cat("current_rms"),
        current_waveform=_cat("current_waveform"),
        fs=backend.fs,
    )
    return AcquisitionResult(
        record=record,
        aborted=aborted,
        abort_reason=reason,
        exposure_s=exposure,
        peak_seal_temp_c=peak,
        seal_temp_trace=temps,
        n_blocks=n_blocks,
    )

def moving_rms(x: np.ndarray, fs: float, win_s: float = 0.02) -> np.ndarray:
    """RMS envelope of a current waveform.

    Datasets store raw current, but the trip path and the current-level features are
    defined on an RMS trajectory, so it is derived rather than the waveform being
    passed where an envelope is expected.

    Lives here rather than in a script because three experiment scripts need it, and
    the last time a helper was copied between them the copies drifted.
    """
    n = max(int(win_s * fs), 1)
    kernel = np.ones(n) / n
    return np.sqrt(np.convolve(np.asarray(x, dtype=float) ** 2, kernel, mode="same"))
