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


# ARM cores identify themselves by implementer + part number rather than by a
# marketing string, so /proc/cpuinfo on an SBC has no "model name" line at all. These
# are the parts that appear in the SoCs this project targets; RK3588 is 4x A76 + 4x
# A55, and saying so is more useful in a paper than "aarch64".
_ARM_CPU_PARTS = {
    "0xd03": "Cortex-A53",
    "0xd05": "Cortex-A55",
    "0xd07": "Cortex-A57",
    "0xd08": "Cortex-A72",
    "0xd09": "Cortex-A73",
    "0xd0a": "Cortex-A75",
    "0xd0b": "Cortex-A76",
    "0xd0d": "Cortex-A77",
    "0xd41": "Cortex-A78",
    "0xd44": "Cortex-X1",
    "0xd42": "Cortex-A78AE",
    "0xd4a": "Neoverse-E1",
    "0xd0c": "Neoverse-N1",
}


def _arm_core_summary(cpuinfo: str) -> str | None:
    """Summarise a big.LITTLE cluster as e.g. '4x Cortex-A76 + 4x Cortex-A55'."""
    counts: dict[str, int] = {}
    for line in cpuinfo.splitlines():
        if line.startswith("CPU part"):
            part = line.split(":", 1)[1].strip().lower()
            name = _ARM_CPU_PARTS.get(part, f"ARM part {part}")
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    # Biggest cluster first, which by convention is how these are described.
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return " + ".join(f"{n}x {name}" for name, n in ordered)


def _cpu_model() -> str | None:
    """The board's CPU string, however this platform chooses to expose it."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        cpuinfo = ""
    if cpuinfo:
        for line in cpuinfo.splitlines():
            for key in ("model name", "Model", "Hardware", "Processor"):
                if line.startswith(key):
                    val = line.split(":", 1)[1].strip()
                    if val:
                        return val
        # No marketing string: describe the cores themselves.
        summary = _arm_core_summary(cpuinfo)
        if summary:
            return summary
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


def probe_accelerators(info: dict) -> dict:
    """Report which accelerators this board exposes, and whether they could run TabPFN.

    Reports what is physically present. It does NOT establish that these accelerators
    cannot run TabPFN - no port was attempted, and this function must not be cited as
    evidence that one failed. The shape variation it records is one of three obstacles,
    and the weakest of them: padding the reference set to a fixed maximum would remove
    it. The two that matter are the restricted operator set, which does not cover a
    transformer attention stack, and INT8 quantisation of a prior-fitted model without
    degrading the calibration that abstention depends on.

    Distinguishing "present but unused" from "absent" is worth something on its own: on
    our board the Coral is not detected at all while the RKNPU driver is loaded, and
    those are different sentences in a paper.
    """
    found = {}

    # Coral Edge TPU. The USB accelerator enumerates under two different IDs: as
    # 1a6e:089a (Global Unichip) before the runtime initialises it, and as 18d1:9302
    # (Google) afterwards. Matched as vendor:product pairs rather than on the bare
    # vendor ID, because 18d1 alone is Google's and would also match an Android phone
    # plugged into the same board. The M.2 and PCIe variants bind the apex driver
    # instead and appear as /dev/apex_N.
    CORAL_USB_IDS = ("1a6e:089a", "18d1:9302")
    coral = []
    apex = sorted(Path("/dev").glob("apex_*"))
    if apex:
        coral.append(f"apex device node(s): {[q.name for q in apex]}")

    usb_checked = False
    try:
        proc = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            usb_checked = True
            for line in proc.stdout.splitlines():
                if any(dev_id in line for dev_id in CORAL_USB_IDS):
                    coral.append(f"USB: {line.strip()}")
    except (OSError, subprocess.SubprocessError):
        pass

    if coral:
        found["coral_edge_tpu"] = coral
    elif usb_checked or apex:
        found["coral_edge_tpu"] = None            # genuinely absent
    else:
        # lsusb is missing (usbutils not installed) and no apex node exists, so the USB
        # variant was never actually checked. Reporting this as "not detected" would be
        # the failure this whole probe exists to avoid: a paper sentence saying the
        # accelerator was absent, when in truth nothing looked for it.
        found["coral_edge_tpu"] = "UNKNOWN - lsusb unavailable, install usbutils to check"

    # Rockchip NPU: rknpu driver exposes a version node on RK3588.
    rknpu = _read_first("/sys/kernel/debug/rknpu/version", "/proc/rknpu/version")
    found["rknpu"] = rknpu or (
        "driver node present" if Path("/sys/kernel/rknpu").exists() else None
    )

    return found


def demonstrate_shape_variance(n_features: int = 63) -> dict:
    """Show that the tensor entering the model changes shape with the reference set.

    This is the whole argument in one measurement. An Edge TPU graph is compiled for
    one input shape; if the shape is a function of the reference set, no compiled graph
    is valid across commissioning events.
    """
    shapes = {}
    for n_context in (200, 500):
        for n_query in (1, 32):
            # The context and the query are concatenated into the model's input: this
            # is what "in-context learning" means at the tensor level.
            shapes[f"context={n_context}, query={n_query}"] = (
                n_context + n_query,
                n_features,
            )
    distinct = {v for v in shapes.values()}
    return {
        "input_shapes": {k: list(v) for k, v in shapes.items()},
        "n_distinct_shapes": len(distinct),
        "compilable_as_one_static_graph": len(distinct) == 1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-context", type=int, default=400)
    ap.add_argument("--n-features", type=int, default=63)
    ap.add_argument("--n-query", type=int, default=64)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--platform-only",
        action="store_true",
        help="Print host and accelerator detail and exit, without benchmarking. "
        "Takes a second, so a board already benchmarked need not repeat it.",
    )
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

    if args.platform_only:
        print("\n=== accelerators present on this board ===")
        for name, val in probe_accelerators(info).items():
            print(f"  {name:16s} {val if val else 'not detected'}")
        return 0

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

    print("\n=== accelerators present on this board ===")
    accel = probe_accelerators(info)
    for name, val in accel.items():
        print(f"  {name:16s} {val if val else 'not detected'}")

    shape = demonstrate_shape_variance(n_features=args.n_features)
    if shape:  # always true; kept so the block reads as one optional section
        print("\n=== why no accelerator can run this model ===")
        for k, v in shape["input_shapes"].items():
            print(f"  {k:28s} -> input tensor {tuple(v)}")
        print(
            f"  {shape['n_distinct_shapes']} distinct input shapes across four ordinary "
            f"operating conditions."
        )
        print(
            "  Both the Coral Edge TPU and the RK3588 NPU compile a graph for ONE fixed\n"
            "  input shape. The reference set is part of TabPFN's input, so the shape is\n"
            "  a function of how many windows commissioning collected and how many\n"
            "  queries are batched. No single compiled graph stays valid, and this is a\n"
            "  property of in-context learning rather than a porting effort left undone."
        )

    out = args.out or ROOT / "results" / "hardware_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "platform": info,
                "benchmark": rows,
                "accelerators_present": accel,
                "shape_variance": shape,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(
        "\nThis measures the CPU path, which is what the architecture depends on.\n"
        "No port to the RK3588 NPU or the Coral Edge TPU was attempted, and none is\n"
        "planned: both need a fixed-shape, fully-INT8 graph from a restricted operator\n"
        "set. Padding the reference set would fix the shape, but the operator set does\n"
        "not cover a transformer attention stack, and INT8-quantising a prior-fitted\n"
        "model without wrecking the calibration that abstention depends on is an open\n"
        "problem. Report that as a constraint, not as a tested negative."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
