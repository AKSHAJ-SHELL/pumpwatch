#!/usr/bin/env python3
"""Rig data collection with a live dry-run seal-temperature interlock.

Acquires in short blocks and re-checks the seal interlock between every one. The
previous version took a single seal temperature as a command-line argument, checked
it once before writing metadata, and left the arrays empty for "DAQ integration" —
which is not a safety system. The temperature that destroys a seal is the one
reached partway through an acquisition that has already started.

Backends: `--backend simulated` runs the whole path, including the abort branch,
with no hardware. Add a real device by implementing `node.daq.DAQBackend` (a block
read plus a seal-temperature read) and registering it in BACKENDS below; nothing in
the control flow needs to change.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pumpwatch.datasets.ownrig import (
    OWNRIG_CONDITIONS,
    OwnRigSessionMeta,
    SealTempCutoff,
    now_utc_iso,
    save_session,
)
from pumpwatch.node.daq import SimulatedDAQ, collect_session


def make_simulated(args) -> SimulatedDAQ:
    return SimulatedDAQ(
        condition=args.condition,
        severity=args.severity,
        rpm=args.rpm,
        n_vanes=args.n_vanes or 6,
        fs=args.fs,
        ambient_temp_c=args.ambient_temp,
        seed=args.seed,
    )


BACKENDS = {"simulated": make_simulated}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--pump-id", required=True)
    p.add_argument("--impeller-id", required=True)
    p.add_argument("--bearing-id", required=True)
    p.add_argument("--mounting", default="stud", choices=["stud", "adhesive", "magnet"])
    p.add_argument("--condition", required=True, choices=OWNRIG_CONDITIONS)
    p.add_argument("--severity", type=float, default=0.0)
    p.add_argument("--rpm", type=float, required=True)
    p.add_argument("--suction-valve", type=float, default=100.0)
    p.add_argument("--discharge-valve", type=float, default=100.0)
    p.add_argument("--ambient-temp", type=float, required=True)
    p.add_argument("--n-vanes", type=int, default=None)
    p.add_argument("--notes", default="")

    p.add_argument("--backend", default="simulated", choices=sorted(BACKENDS))
    p.add_argument("--fs", type=float, default=26_700.0)
    p.add_argument("--duration-s", type=float, default=5.0)
    p.add_argument("--block-s", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-seal-temp", type=float, default=80.0,
        help="Hard interlock on seal-face temperature (dry-run sessions).",
    )
    p.add_argument(
        "--max-exposure-s", type=float, default=20.0,
        help="Hard limit on dry-run exposure regardless of temperature.",
    )
    p.add_argument(
        "--real-time", action="store_true",
        help="Sleep between blocks. Off by default so the simulated backend is fast.",
    )
    args = p.parse_args()

    cutoff = SealTempCutoff(
        max_seal_temp_c=args.max_seal_temp, max_exposure_s=args.max_exposure_s
    )
    meta = OwnRigSessionMeta(
        session_id=args.session_id,
        pump_id=args.pump_id,
        impeller_id=args.impeller_id,
        bearing_id=args.bearing_id,
        mounting_type=args.mounting,
        condition=args.condition,
        severity=args.severity,
        rpm=args.rpm,
        suction_valve_pct=args.suction_valve,
        discharge_valve_pct=args.discharge_valve,
        ambient_temp_c=args.ambient_temp,
        # Placeholder; collect_session overwrites this with the peak actually
        # measured, so the stored metadata is an observation, not an intention.
        seal_temp_c=args.ambient_temp,
        timestamp_utc=now_utc_iso(),
        n_vanes=args.n_vanes,
        notes=args.notes,
    )

    def stop_pump(reason: str) -> None:
        # Wire to the contactor / VFD stop here. Printed to stderr so it is visible
        # even when stdout is being captured.
        print(f"!! STOP PUMP: {reason}", file=sys.stderr)

    backend = BACKENDS[args.backend](args)
    try:
        result = collect_session(
            backend,
            meta,
            duration_s=args.duration_s,
            cutoff=cutoff,
            block_s=args.block_s,
            stop_pump=stop_pump,
            sleep=time.sleep if args.real_time else (lambda s: None),
        )
    finally:
        backend.close()

    path = save_session(args.root, result.record)
    sidecar = Path(args.root) / f"{args.session_id}_acquisition.json"
    sidecar.write_text(
        json.dumps(
            {
                "aborted": result.aborted,
                "abort_reason": result.abort_reason,
                "exposure_s": result.exposure_s,
                "peak_seal_temp_c": result.peak_seal_temp_c,
                "n_blocks": result.n_blocks,
                "backend": args.backend,
                "fs": backend.fs,
                "seal_temp_trace": result.seal_temp_trace,
                "cutoff": {
                    "max_seal_temp_c": cutoff.max_seal_temp_c,
                    "max_exposure_s": cutoff.max_exposure_s,
                },
                "synthetic": args.backend == "simulated",
            },
            indent=2,
        )
    )
    print(f"wrote {path}")
    print(f"wrote {sidecar}")
    print(
        f"exposure={result.exposure_s:.2f}s peak_seal={result.peak_seal_temp_c:.1f}C "
        f"blocks={result.n_blocks} aborted={result.aborted}"
    )
    if result.aborted:
        print(f"ABORTED: {result.abort_reason}", file=sys.stderr)
        # Non-zero so a collection harness or shell loop notices.
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
