# Running the gateway on the OrangePi

Everything here is a command to run on the board. Nothing in this file has been
executed — the repository has no access to the hardware — so treat every output
below as "what you should see", not as a result.

The reason to do this at all: **every latency number in the results was measured on
a laptop.** That is what makes "RK3588 gateway" a claim about hardware nobody
benchmarked, and it is the largest remaining hardware gap in the paper. One command
closes it.

---

## 1. Identify the board (30 seconds)

```bash
cat /proc/device-tree/model
uname -m                 # expect aarch64
nproc
free -g
```

`make bench-hardware` reads all of this itself, so this step is only to know in
advance what you are dealing with. What matters is the SoC: **RK3588 or RK3588S**
gives you the 16 GB-capable board the design assumes. Anything else still works —
just describe it accurately in the paper as "an ARM-class gateway" rather than
naming an SoC you did not test.

## 2. Install (the step most likely to cost you time)

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev build-essential
git clone <your repo> pump_monitoring && cd pump_monitoring
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install -e ".[tabpfn]"
```

**If torch will not install:** this is the common failure and it is not worth a day.
PyPI ships `aarch64` wheels for recent torch, but distribution Python versions on
SBC images are often ahead of or behind what has wheels. In order of preference:

1. `pip install torch --index-url https://download.pytorch.org/whl/cpu`
2. Use a Python version with wheels (3.10–3.12 are safest); `pyenv` or a container.
3. **Give up and say so.** Restate the gateway as "an ARM-class single-board
   computer" in the paper, keep the laptop latencies clearly labelled as laptop
   latencies, and spend the time on §2 Related work instead. A missing hardware
   measurement, honestly labelled, costs you far less than a day lost to
   cross-compiling torch.

Building torch from source on this board is **not** a reasonable use of nine days.

## 3. Run the benchmark

```bash
make bench-hardware
```

This writes `results/hardware_bench.json`, stamped with the board string, CPU, RAM,
core count and thread setting — a latency figure without the machine and the thread
count is not reproducible, so the script refuses to produce one.

It reports three things:

- **Inference latency** in four configurations — cached and uncached KV, one and
  eight ensemble members — as seconds per batch and milliseconds per window.
- **The two design claims, measured on your board.** On the development laptop the
  KV cache is worth 7.3× and the 8-member ensemble costs 5.5×. If your board
  reproduces the *ordering* the design claim holds; the magnitudes will differ.
- **A warning if there is no device tree**, because a laptop run of this script
  otherwise looks exactly like a board run.

Everything is single-threaded on purpose. torch and LightGBM each ship an OpenMP
runtime and crash the process together, and single-threaded is the honest gateway
configuration anyway — it costs about 2.5× against unpinned laptop numbers.

## 4. Optionally, re-run one experiment on the board

```bash
make experiment-espset          # not experiment-espset-full — that is ~40 min on a laptop
```

Only worth it if step 3 succeeded and you have time. It changes no scientific
conclusion — the same code produced the laptop numbers — but it lets you say the
cross-machine result was reproduced on the deployment hardware. If it is slow,
that *is* the finding, and it belongs in the paper.

---

## The Coral TPU: do not attempt it

You have one, the synopsis proposed it, and it cannot run this model. This is worth
stating precisely because "we didn't get to it" and "it cannot work" are different
claims and only the second is a contribution.

`make bench-hardware` prints the demonstration. The argument in one line: **the
reference set is part of TabPFN's input**, so the tensor entering the model has
shape `(n_context + n_query, n_features)`. Across four ordinary operating
conditions — 200 or 500 reference windows, 1 or 32 queries — that is four different
input shapes:

| Condition | Input tensor |
|---|---|
| 200 reference windows, 1 query | (201, 63) |
| 200 reference windows, 32 queries | (232, 63) |
| 500 reference windows, 1 query | (501, 63) |
| 500 reference windows, 32 queries | (532, 63) |

The Edge TPU compiler produces a graph for **one** fixed input shape, from a
restricted INT8 operator set. The RK3588 NPU's RKNN toolkit has the same
requirement. There is no single graph to compile, and the shape varies precisely
because of the mechanism the whole paper is about — commissioning by substituting a
reference set changes the input.

So: report it as a negative result with the shapes above as evidence. It is one
paragraph, it is honest, and it is more useful to a reader than silence. What it is
**not** is a reason to abandon the architecture — the gateway is a shared,
mains-powered board where CPU inference is affordable. The accelerator was only ever
proposed as an optimisation.

If you want to check the hardware is at least present and working — separate from
whether it can run TabPFN — the probe in `make bench-hardware` looks for `apex`
device nodes and the USB vendor IDs, and reports what it finds.

---

## The MCU node: nothing to run here

The node tier is an energy and algorithm model in this repository, not firmware.
`node/acquire.py`, `node/energy.py` and `node/airtime.py` are explicitly labelled
design models, and `node/gates.py` and `node/trip.py` are the gate and trip logic
evaluated offline against real data. There is no MCU build to flash, and the paper
does not claim one. The OrangePi is the gateway tier only.

## Checklist

- [ ] `cat /proc/device-tree/model` — know what you have
- [ ] `pip install -e ".[tabpfn]"` — time-boxed; abandon if torch fights you
- [ ] `make bench-hardware` — the one command that matters
- [ ] Copy the measured numbers into §3.4, replacing the laptop figures
- [ ] Paste the shape table above into the Coral paragraph
- [ ] ❌ Do not attempt an Edge TPU or RKNN port
