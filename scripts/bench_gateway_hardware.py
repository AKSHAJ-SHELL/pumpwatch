#!/usr/bin/env python3
"""Measure gateway inference latency on the machine it actually runs on.

Every latency number in the results so far was measured on a laptop, which makes
"RK3588 gateway" a claim about hardware nobody benchmarked. This script closes that
gap: run it on the target board and it writes a results file the paper can cite,
stamped with enough platform detail that a reader can tell which machine produced it.

    python scripts/bench_gateway_hardware.py

It records the board identity (/proc/device-tree/model on ARM SBCs), CPU, core count,
RAM and thread settings alongside the measurement, because a latency figure without
the machine and the thread count is not reproducible.

On the accelerator question: do not expect to move this to the NPU or a Coral TPU.
TabPFN's input shape varies by construction - the context is part of the input - and
both the RK3588 NPU (RKNN) and the Coral Edge TPU require statically-shaped, INT8
graphs built from a supported op set. Measuring honest ARM CPU latency is the
achievable result; an accelerated one is not, and the script says so rather than
leaving a reader to assume the NPU was simply never tried.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _read_first(*paths: str) -> str | None:
    for p in paths:
        try:
            return Path(p).read_text(errors="ignore").strip("\x00 \n\t")
        except OSError:
            continue
    return None


def _cpu_model() -> str | None:
    """The board's CPU string, however this platform chooses to expose it."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            for key in ("model name", "Model", "Hardware", "Processor"):
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return None


def _total_ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return round(int(out) / 1024**3, 1) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def describe_platform() -> dict:
    """Identify the machine well enough that the measurement is attributable."""
    board = _read_first("/proc/device-tree/model", "/sys/firmware/devicetree/base/model")
    return {
        # Present on ARM SBCs (OrangePi, Raspberry Pi); absent on x86 and macOS.
        "board": board,
        "is_arm_sbc": board is not None,
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
        "python": platform.python_version(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-context", type=int, default=400)
    ap.add_argument("--n-features", type=int, default=63)
    ap.add_argument("--n-query", type=int, default=64)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    warnings.filterwarnings("ignore", message=".*OOD abstention disabled.*")

    info = describe_platform()
    print("=== gateway host ===")
    for k, v in info.items():
        if v is not None:
            print(f"  {k:18s} {v}")
    if not info["is_arm_sbc"]:
        print(
            "\n  Note: no device-tree model, so this is not an ARM SBC. The numbers\n"
            "  below are valid for THIS machine; they do not substitute for a board\n"
            "  measurement, which is the entire point of running this script."
        )

    try:
        # Imported first and held to one thread, matching baselines.make_lightgbm:
        # torch and LightGBM each ship an OpenMP runtime and crash the process
        # together. Single-threaded is also the honest gateway measurement.
        import torch

        torch.set_num_threads(1)
        from pumpwatch.gateway.tabpfn_clf import ATTRIBUTION_NOTICE, benchmark_tabpfn
    except ImportError as exc:
        print(f"\nTabPFN or torch not installed on this host: {exc}")
        print('Install with: pip install -e ".[tabpfn]"')
        return 1

    print(f"\n=== inference latency ({ATTRIBUTION_NOTICE}) ===")
    rows = benchmark_tabpfn(
        n_context=args.n_context,
        n_features=args.n_features,
        n_query=args.n_query,
        n_repeats=args.n_repeats,
    )
    print(f"  {'fit_mode':22s} {'n_est':>5s} {'fit_s':>8s} {'predict_s':>10s} {'ms/window':>10s}")
    for r in rows:
        per_window_ms = 1000.0 * r["predict_latency_s"] / max(r["n_query"], 1)
        r["ms_per_window"] = per_window_ms
        print(
            f"  {r['fit_mode']:22s} {r['n_estimators']:5d} {r['fit_latency_s']:8.3f} "
            f"{r['predict_latency_s']:10.3f} {per_window_ms:10.2f}"
        )

    # The design's two claimed wins, restated as this machine measured them.
    by_key = {(r["fit_mode"], r["n_estimators"]): r["predict_latency_s"] for r in rows}
    cached = by_key.get(("fit_with_cache", 1))
    uncached = by_key.get(("fit_preprocessors", 1))
    if cached and uncached:
        print(f"\n  KV cache speedup at n_estimators=1: {uncached / cached:.1f}x")
    single = by_key.get(("fit_with_cache", 1))
    ensemble = by_key.get(("fit_with_cache", 8))
    if single and ensemble:
        print(f"  Cost of the 8-member ensemble:     {ensemble / single:.1f}x")

    out = args.out or ROOT / "results" / "hardware_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"platform": info, "benchmark": rows}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(
        "\nThis measures the CPU path. The RK3588 NPU and the Coral Edge TPU cannot\n"
        "run TabPFN: its input shape varies by construction, and both accelerators\n"
        "require statically-shaped INT8 graphs. That is a property of the model, not\n"
        "a porting effort left undone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
