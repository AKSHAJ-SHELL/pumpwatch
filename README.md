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

TabPFN is optional (`pip install -e ".[tabpfn]"`). Pin **v2** for commercial
use — `pip install tabpfn` may give v3 (research-only).

## Commands

```bash
make test
make figures
make experiment          # synth/demo Twente + LOMO baselines
make experiment-tabpfn   # also runs TabPFN if installed
```

## Profiles

- `full` — vibration + CT (surface pumps)
- `ct_only` — CT only (submersible borewell — dominant smallholder case)

See `configs/profiles.yaml`.
