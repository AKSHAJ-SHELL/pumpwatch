#!/usr/bin/env python3
"""Rig data collection helper with dry-run seal-temperature cutoff.

This is a schema/safety harness — wire it to your DAQ. It refuses to continue
a dry-run session past SealTempCutoff limits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pumpwatch.datasets.ownrig import (
    OWNRIG_CONDITIONS,
    OwnRigRecord,
    OwnRigSessionMeta,
    SealTempCutoff,
    now_utc_iso,
    save_session,
)


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
    p.add_argument("--seal-temp", type=float, required=True)
    p.add_argument("--n-vanes", type=int, default=None)
    p.add_argument("--exposure-s", type=float, default=0.0, help="dry-run exposure so far")
    p.add_argument("--max-seal-temp", type=float, default=80.0)
    p.add_argument("--max-exposure-s", type=float, default=20.0)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    cutoff = SealTempCutoff(max_seal_temp_c=args.max_seal_temp, max_exposure_s=args.max_exposure_s)
    if args.condition == "dry_run" and cutoff.should_abort(args.seal_temp, args.exposure_s):
        print(
            f"ABORT dry-run: seal_temp={args.seal_temp}C exposure={args.exposure_s}s "
            f"exceeds cutoff ({cutoff.max_seal_temp_c}C / {cutoff.max_exposure_s}s)",
            file=sys.stderr,
        )
        sys.exit(2)

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
        seal_temp_c=args.seal_temp,
        timestamp_utc=now_utc_iso(),
        n_vanes=args.n_vanes,
        notes=args.notes,
    )
    # Arrays filled by DAQ integration — metadata-only write is valid for dry-run.
    path = save_session(args.root, OwnRigRecord(meta=meta))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
