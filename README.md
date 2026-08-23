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

This code is MIT ([LICENSE](LICENSE)). It is **Built with PriorLabs-TabPFN**, and
both evaluation datasets are CC BY 4.0 — [ATTRIBUTION.md](ATTRIBUTION.md) collects
the three sets of obligations and their citations in one place. No dataset files
are in this repository; every loader raises with download instructions instead.

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

## Datasets

| Result file | Data | Cite as a result? |
|---|---|---|
| `results_espset_*.json` | **Real** — ESPset, 11 in-service submersible pumps | ✅ |
| `results_twente_real.json` | **Real** — Twente/4TU rig, 2 motors | ✅ |
| `results_full.json` | Synthetic stand-in (`data/twente_demo/`) | ❌ |

`data/twente_demo/` is generated, and its fault signatures were written into the
generator by hand. Those scores verify that the feature pipeline and splits recover
signatures known to be present — a wiring check and an upper bound, not evidence
about pumps.

**Getting the real data** (both are gitignored; loaders print instructions if absent):

```bash
# ESPset — ~115 MB, CC BY 4.0, doi.org/10.17632/m268jsw339.3
#   extract features.csv and spectrum.csv into data/espset/
make experiment-espset

# Twente/4TU — 20.8 GB, CC BY 4.0, nested 7z
#   doi.org/10.4121/2b61183e-c14f-4131-829b-cc4822c369d0
#   extract into data/raw/twente_sel/  (full extraction needs ~320 GB; a subset works)
make experiment-twente
```

The two are complementary and neither alone is sufficient: **ESPset is the only
source that supports leave-one-machine-out** (Twente's two motors share no fault
class), and **Twente is the only source with a current channel**, so it is the only
place `ct_only` can be tested on real signals. See [DESIGN.md](DESIGN.md) §−2.

## Rig collection

```bash
make rig-demo   # runs the dry-run collection path, including the interlock abort
```

The seal interlock is re-checked between acquisition blocks, stops the pump on
breach, and keeps the partial recording. `--backend simulated` needs no hardware;
add a real device by implementing `node.daq.DAQBackend`.

## Gotcha: OpenMP

LightGBM and torch each ship an OpenMP runtime and segfault the process together on
macOS. `scripts/run_experiment.py` sets `OMP_NUM_THREADS=1` before any import for
this reason; reported latencies are single-threaded and therefore conservative.
