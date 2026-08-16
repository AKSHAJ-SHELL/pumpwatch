# PumpWatch research codebase

Two-tier irrigation pump fault monitoring: MCU gating + local dry-run trip,
gateway in-context classification (TabPFN v2 pinned) with honest baselines.

Read [DESIGN.md](DESIGN.md) before changing architecture.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

TabPFN is optional (`pip install -e ".[tabpfn]"`), pinned to `tabpfn>=2.0,<3`.
That package constraint **is** the v2 model pin — the 2.x line has no
`ModelVersion` selector, and there is no 3.x package (PyPI jumps 2.x → 6.x).
tabpfn 2.2.1 carries the Prior Labs License (Apache 2.0 + attribution); see
[DESIGN.md](DESIGN.md) §10 for the attribution obligation that comes with it.

## Commands

```bash
make test
make experiment          # demo Twente + gate + full leakage ladder + baselines
make experiment-tabpfn   # also runs TabPFN if installed
make figures             # reads results/*.json — run an experiment first
```

`make figures` fails if `results/` is empty. Result figures are built from measured
numbers only; nothing synthesises a placeholder score.

## Profiles

- `full` — vibration + CT (surface pumps)
- `ct_only` — CT only (submersible borewell — dominant smallholder case)

See `configs/profiles.yaml`.

## ⚠️ The numbers in `results/` are synthetic

`data/twente_demo/` is a generated stand-in, not the real 4TU dataset, and its fault
signatures were written into the generator by hand. The scores verify that the
feature pipeline and the splits recover signatures known to be present — a wiring
check and an upper bound. They are **not** evidence about real pumps and must not be
cited as results. See [DESIGN.md](DESIGN.md) §4.

## Gotcha: OpenMP

LightGBM and torch each ship an OpenMP runtime and segfault the process together on
macOS. `scripts/run_experiment.py` sets `OMP_NUM_THREADS=1` before any import for
this reason; reported latencies are single-threaded and therefore conservative.
