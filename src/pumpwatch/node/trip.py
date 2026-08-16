"""Dry-run under-current trip path — commodity-relay equivalent at the MCU.

This is NOT an ML classifier class. Detection terminates at the node with a
trip decision. Closed-valve confusers are first-class: false-trip analysis
is the figure that replaces a naked CUSUM plot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pumpwatch.node.gates import CUSUM1D
from pumpwatch.physics import DryRunCurrentParams, dry_run_current
from pumpwatch.synth import Condition, PumpMeta, SynthConfig, generate_record


@dataclass
class TripConfig:
    """Parameters for the under-current dry-run trip."""

    cusum_k: float = 0.5
    cusum_h: float = 5.0
    # Hard absolute floor as backstop (fraction of rated). Kept below
    # closed-valve (~0.70) so valve throttling alone should not trip it.
    absolute_floor_fraction: float = 0.55
    # Require persistence: N consecutive candidate samples before actuating
    persistence_n: int = 5
    # Simulated actuation latency after decision (contactor)
    actuation_latency_s: float = 0.05
    sample_period_s: float = 0.05  # how often RMS is evaluated while pump runs


@dataclass
class TripDecision:
    detected: bool
    detection_time_s: Optional[float]
    trip_time_s: Optional[float]  # detection + actuation latency
    detection_delay_s: Optional[float]  # relative to fault onset
    false_trip: bool = False
    scores: list[float] = field(default_factory=list)


@dataclass
class DryRunTrip:
    """One-sided CUSUM on motor current + absolute floor backstop."""

    config: TripConfig = field(default_factory=TripConfig)
    cusum: CUSUM1D = field(default_factory=CUSUM1D)
    rated_current_a: float = 10.0
    _persist: int = 0

    def fit(self, healthy_current_rms: np.ndarray, rated_current_a: float) -> "DryRunTrip":
        self.rated_current_a = rated_current_a
        self.cusum = CUSUM1D(
            k=self.config.cusum_k,
            h=self.config.cusum_h,
            direction="down",
        )
        self.cusum.fit(healthy_current_rms)
        self._persist = 0
        return self

    def reset(self) -> None:
        self.cusum.reset()
        self._persist = 0

    def update(self, current_rms: float, t: float) -> TripDecision:
        """Feed one RMS sample; return decision (may not yet trip)."""
        hit, score = self.cusum.update(current_rms)
        floor_hit = current_rms < self.config.absolute_floor_fraction * self.rated_current_a
        candidate = hit or floor_hit
        if candidate:
            self._persist += 1
        else:
            self._persist = 0

        if self._persist >= self.config.persistence_n:
            det_t = t
            trip_t = t + self.config.actuation_latency_s
            return TripDecision(
                detected=True,
                detection_time_s=det_t,
                trip_time_s=trip_t,
                detection_delay_s=None,  # caller fills vs onset
                scores=[score],
            )
        return TripDecision(
            detected=False,
            detection_time_s=None,
            trip_time_s=None,
            detection_delay_s=None,
            scores=[score],
        )

    def run_trajectory(
        self,
        t: np.ndarray,
        current_rms: np.ndarray,
        onset_s: Optional[float] = None,
        expect_fault: bool = True,
    ) -> TripDecision:
        """Run the full trajectory; return first trip decision."""
        self.reset()
        # Subsample at sample_period_s
        dt = self.config.sample_period_s
        t0, t1 = float(t[0]), float(t[-1])
        sample_times = np.arange(t0, t1, dt)
        scores: list[float] = []
        for ts in sample_times:
            idx = int(np.searchsorted(t, ts, side="right") - 1)
            idx = max(0, min(idx, len(t) - 1))
            decision = self.update(float(current_rms[idx]), float(ts))
            scores.extend(decision.scores)
            if decision.detected:
                delay = None
                if onset_s is not None and decision.detection_time_s is not None:
                    delay = decision.detection_time_s - onset_s
                false_trip = (not expect_fault) or (
                    onset_s is not None and decision.detection_time_s is not None and decision.detection_time_s < onset_s
                )
                return TripDecision(
                    detected=True,
                    detection_time_s=decision.detection_time_s,
                    trip_time_s=decision.trip_time_s,
                    detection_delay_s=delay,
                    false_trip=false_trip,
                    scores=scores,
                )
        return TripDecision(
            detected=False,
            detection_time_s=None,
            trip_time_s=None,
            detection_delay_s=None,
            false_trip=False,
            scores=scores,
        )


@dataclass
class FalseTripAnalysis:
    """Compare dry-run detection against closed-valve confusers."""

    dry_run_detection_rate: float
    dry_run_median_delay_s: float
    closed_valve_false_trip_rate: float
    healthy_false_trip_rate: float
    delays_s: list[float]


def evaluate_trip_path(
    n_trials: int = 40,
    seed: int = 0,
    trip_config: Optional[TripConfig] = None,
    duration_s: float = 8.0,
) -> FalseTripAnalysis:
    """Monte Carlo: dry-run true positives vs closed-valve / healthy false trips."""
    rng = np.random.default_rng(seed)
    cfg = trip_config or TripConfig()
    meta = PumpMeta(rated_current_a=10.0)

    # Fit on healthy
    healthy_samples = []
    for i in range(30):
        rec = generate_record(
            Condition.HEALTHY,
            meta=meta,
            config=SynthConfig(duration_s=2.0, seed=int(rng.integers(0, 1e9)), noise_std=0.02),
            rate="lo",
        )
        # Decimate current RMS to sample_period
        step = max(1, int(cfg.sample_period_s * rec.fs))
        healthy_samples.append(rec.current_rms[::step])
    healthy_all = np.concatenate(healthy_samples)

    trip = DryRunTrip(config=cfg)
    trip.fit(healthy_all, rated_current_a=meta.rated_current_a)

    delays: list[float] = []
    dry_hits = 0
    cv_trips = 0
    healthy_trips = 0

    for i in range(n_trials):
        onset = float(rng.uniform(1.0, 2.0))
        # Dry run
        params = DryRunCurrentParams(rated_current_a=10.0, noise_std_fraction=0.02)
        t = np.arange(0.0, duration_s, cfg.sample_period_s)
        i_dry = dry_run_current(t, onset, params, "dry_run", rng=rng)
        d = trip.run_trajectory(t, i_dry, onset_s=onset, expect_fault=True)
        if d.detected and not d.false_trip:
            dry_hits += 1
            if d.detection_delay_s is not None:
                delays.append(d.detection_delay_s)

        # Closed valve confuser
        i_cv = dry_run_current(t, onset, params, "closed_valve", rng=rng)
        d_cv = trip.run_trajectory(t, i_cv, onset_s=onset, expect_fault=False)
        if d_cv.detected:
            cv_trips += 1

        # Healthy
        i_h = dry_run_current(t, onset, params, "healthy", rng=rng)
        d_h = trip.run_trajectory(t, i_h, onset_s=None, expect_fault=False)
        if d_h.detected:
            healthy_trips += 1

    return FalseTripAnalysis(
        dry_run_detection_rate=dry_hits / n_trials,
        dry_run_median_delay_s=float(np.median(delays)) if delays else float("nan"),
        closed_valve_false_trip_rate=cv_trips / n_trials,
        healthy_false_trip_rate=healthy_trips / n_trials,
        delays_s=delays,
    )
