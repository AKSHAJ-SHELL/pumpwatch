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


@dataclass
class TripConfig:
    """Parameters for the under-current dry-run trip.

    Three mechanisms, and each one has to matter:

    * **CUSUM** detects the abrupt downward shift with minimal expected delay
      (Page 1954). On its own it fires on *any* load loss, including a throttled
      discharge valve.
    * **Absolute floor** discriminates by depth. Dry running drops current to
      ~45% of rated; a closed discharge valve only to ~70%. A floor between the
      two is what separates the fault from its most common confuser.
    * **Persistence** rejects transient dips.

    These are combined with AND, not OR. Under OR the floor was decorative: CUSUM
    fired on the closed-valve drop long before the level mattered, and the trip
    path fired on 100% of closed-valve confusers — which is a contactor that shuts
    off irrigation every time a farmer throttles a valve.
    """

    cusum_k: float = 0.5
    cusum_h: float = 5.0
    # Hard absolute floor (fraction of rated). Sits between dry-run (~0.45) and
    # closed-valve (~0.70) so valve throttling alone must not trip it.
    absolute_floor_fraction: float = 0.55
    # Require persistence: N consecutive candidate samples before actuating
    persistence_n: int = 5
    # Simulated actuation latency after decision (contactor)
    actuation_latency_s: float = 0.05
    sample_period_s: float = 0.05  # how often RMS is evaluated while pump runs
    # AND the depth check with the CUSUM shift detector. Setting this False
    # restores the OR behaviour and is retained only to reproduce the failure.
    require_floor: bool = True


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
        # AND: the current must both have shifted abruptly and be deep enough to be
        # dry running rather than a throttled valve. See TripConfig.
        candidate = (hit and floor_hit) if self.config.require_floor else (hit or floor_hit)
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


# Real motor current on an irrigation pump is not quiet: suction head varies with
# well drawdown, the supply sags under rural load-shedding, and flow is unsteady.
# The original 2% figure made a 30% closed-valve drop a ~15-sigma excursion, which
# is why CUSUM fired on every confuser. Treat this as a parameter to sweep, not a
# constant to trust — it is the single assumption the false-trip rate is most
# sensitive to, and it must be measured on the rig.
HEALTHY_CURRENT_NOISE_FRACTION = 0.08


def evaluate_trip_path(
    n_trials: int = 40,
    seed: int = 0,
    trip_config: Optional[TripConfig] = None,
    duration_s: float = 8.0,
    healthy_noise_fraction: float = HEALTHY_CURRENT_NOISE_FRACTION,
) -> FalseTripAnalysis:
    """Monte Carlo: dry-run true positives vs closed-valve / healthy false trips."""
    rng = np.random.default_rng(seed)
    cfg = trip_config or TripConfig()
    rated = 10.0
    params = DryRunCurrentParams(
        rated_current_a=rated, noise_std_fraction=healthy_noise_fraction
    )
    t = np.arange(0.0, duration_s, cfg.sample_period_s)

    # Commissioning baseline: healthy current at the node's own sampling rate.
    healthy_all = np.concatenate(
        [
            dry_run_current(t, 0.0, params, "healthy", rng=rng)
            for _ in range(30)
        ]
    )

    trip = DryRunTrip(config=cfg)
    trip.fit(healthy_all, rated_current_a=rated)

    delays: list[float] = []
    dry_hits = 0
    cv_trips = 0
    healthy_trips = 0

    for _ in range(n_trials):
        onset = float(rng.uniform(1.0, 2.0))

        i_dry = dry_run_current(t, onset, params, "dry_run", rng=rng)
        d = trip.run_trajectory(t, i_dry, onset_s=onset, expect_fault=True)
        if d.detected and not d.false_trip:
            dry_hits += 1
            if d.detection_delay_s is not None:
                delays.append(d.detection_delay_s)

        # Closed valve confuser — the discriminator that justifies this path.
        i_cv = dry_run_current(t, onset, params, "closed_valve", rng=rng)
        d_cv = trip.run_trajectory(t, i_cv, onset_s=onset, expect_fault=False)
        if d_cv.detected:
            cv_trips += 1

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


def sweep_trip_operating_points(
    floor_fractions: Optional[list[float]] = None,
    persistence_values: Optional[list[int]] = None,
    cusum_h_values: Optional[list[float]] = None,
    n_trials: int = 40,
    seed: int = 0,
    healthy_noise_fraction: float = HEALTHY_CURRENT_NOISE_FRACTION,
) -> list[dict]:
    """Sweep the trip parameters and return every operating point.

    The operating point is a safety decision with asymmetric costs — a missed
    dry-run destroys a mechanical seal in under a minute, a false trip costs
    irrigation hours — so it must be *chosen* off a measured curve, not hardcoded.
    """
    floor_fractions = floor_fractions or [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
    persistence_values = persistence_values or [1, 2, 3, 5, 8]
    cusum_h_values = cusum_h_values or [3.5, 5.0, 8.0]

    points = []
    for floor in floor_fractions:
        for persist in persistence_values:
            for h in cusum_h_values:
                cfg = TripConfig(
                    cusum_h=h,
                    persistence_n=persist,
                    absolute_floor_fraction=floor,
                )
                res = evaluate_trip_path(
                    n_trials=n_trials,
                    seed=seed,
                    trip_config=cfg,
                    healthy_noise_fraction=healthy_noise_fraction,
                )
                points.append({
                    "absolute_floor_fraction": floor,
                    "persistence_n": persist,
                    "cusum_h": h,
                    "detection_rate": res.dry_run_detection_rate,
                    "closed_valve_false_trip_rate": res.closed_valve_false_trip_rate,
                    "healthy_false_trip_rate": res.healthy_false_trip_rate,
                    "median_delay_s": res.dry_run_median_delay_s,
                })
    return points


def select_operating_point(
    points: list[dict],
    max_false_trip_rate: float = 0.02,
    max_delay_s: float = 30.0,
) -> Optional[dict]:
    """Pick the fastest-detecting point that meets the false-trip and delay budget.

    `max_delay_s` is bounded by seal survival: DESIGN §0.2 puts mechanical seal
    destruction under 60 s of dry running, so detection plus actuation has to land
    well inside that.
    """
    feasible = [
        p for p in points
        if max(p["closed_valve_false_trip_rate"], p["healthy_false_trip_rate"])
        <= max_false_trip_rate
        and np.isfinite(p["median_delay_s"])
        and p["median_delay_s"] <= max_delay_s
    ]
    if not feasible:
        return None
    # Highest detection rate, then fastest.
    return max(feasible, key=lambda p: (p["detection_rate"], -p["median_delay_s"]))
