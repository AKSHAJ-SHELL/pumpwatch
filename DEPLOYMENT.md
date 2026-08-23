# Running the gateway on the OrangePi

Everything here is a command to run on the board. Nothing in this file has been
executed — the repository has no access to the hardware — so treat every output
below as "what you should see", not as a result.

The reason to do this at all: **every latency number in the results was measured on
a laptop.** That is what makes "RK3588 gateway" a claim about hardware nobody
benchmarked, and it is the largest remaining hardware gap in the paper. One command
closes it.

---

## 0. SSH in, and know what you are actually doing

> **Every command below is prefixed with the machine it runs on.**
> `laptop$` means your Mac. `pi$` means the SSH session on the board. Getting this
> wrong is the easiest mistake here — step 1 in particular is a laptop command, and
> running it on the Pi copies the board to itself.
>
> Anything in `<angle brackets>` is a placeholder to replace. Leave one in and bash
> reads `<ip>` as a redirect from a file called `ip`, which is why you get
> "No such file or directory" rather than a useful error.

```bash
laptop$ ssh orangepi@<ip>      # password is often "orangepi" on a stock image
```

**Finding `<ip>`:** on the board, `hostname -I | awk '{print $1}'`. Many Orange Pi
images also run mDNS, in which case `orangepi5plus.local` works in place of the
address everywhere below and you never need the number.

The whole job is **one command** — `make bench-hardware` — and everything before it is
getting the code and its dependencies onto the board. Three facts that shape the plan:

- **The benchmark needs no dataset.** It generates its own context and query matrices,
  so you do not have to move ESPset or Twente onto the board. That is the difference
  between a 20-minute job and an afternoon.
- **The virtualenv cannot be copied.** It is 1.0 GB of x86/macOS-built wheels here;
  torch alone is 529 MB installed. It must be rebuilt on the board from aarch64
  wheels.
- **The repository has no git remote configured.** `git clone` will not work until you
  push it somewhere. `rsync` is the shorter path.

## 1. Get the code onto the board (~11 MB)

⚠️ **This runs on the laptop, not the Pi.** Leave the SSH session open and use a
second terminal window.

```bash
laptop$ rsync -av --exclude .venv --exclude data --exclude .git \
        --exclude results --exclude figures \
        ~/pump_monitoring/ orangepi@<ip>:~/pump_monitoring/
```

⚠️ **`--exclude results` is not optional.** rsync does not read `.gitignore`, so without
it every sync copies the laptop's `results/` over the board's — including a
`hardware_bench.json` from a laptop run, silently replacing the board measurement you
came here to collect. The benchmark now names its output after the machine
(`hardware_bench_rk3588_opi_5_plus.json`) so the two cannot collide even if you forget,
but do not rely on that alone.

The exclusions matter: `.venv` is 1.0 GB of wheels built for the wrong architecture,
and `data/` is up to 20.8 GB and unnecessary for the benchmark. What is left is about
11 MB.

If you would rather use git, push this repository to GitHub first and clone it on the
board — there is currently no remote, so `git clone` has nothing to point at.

## 2. Identify the board (30 seconds)

Back on the Pi:

```bash
pi$ cat /proc/device-tree/model
pi$ uname -m                 # expect aarch64
pi$ nproc
pi$ free -g
```

An **Orange Pi 5 Plus reports RK3588**, which is the SoC the design assumes — that is
the good case, and you can name it in the paper.

`make bench-hardware` reads all of this itself, so this is only to know in advance
what you are dealing with. What matters is the SoC: **RK3588 or RK3588S** is the board
the design assumes. Anything else still works — just describe it accurately in the
paper as "an ARM-class gateway" rather than naming an SoC you did not test.

## 3. Install (the step most likely to cost you time)

```bash
pi$ sudo apt update && sudo apt install -y python3-venv python3-dev build-essential tmux
pi$ cd ~/pump_monitoring
pi$ python3 -m venv .venv && . .venv/bin/activate
pi$ pip install -U pip
pi$ pip install -e ".[tabpfn]"
```

Expect this to take a while and pull several hundred MB — torch is ~529 MB installed.
Use a wired connection if you have one.

**If torch will not install:** this is the common failure and it is not worth a day.
PyPI ships `aarch64` wheels for recent torch, but SBC images often run a Python version
ahead of or behind what has wheels. In order of preference:

1. `pip install torch --index-url https://download.pytorch.org/whl/cpu`
2. Use a Python with wheels (3.10–3.12 are safest) — `pyenv`, or a container.
3. **Give up and say so.** Restate the gateway as "an ARM-class single-board computer"
   in the paper, keep the laptop latencies clearly labelled as laptop latencies, and
   spend the time on §2 Related work instead. A missing hardware measurement, honestly
   labelled, costs far less than a day lost to cross-compiling torch.

Building torch from source on this board is **not** a reasonable use of nine days.

### Model weights: 28 MB, fetched on first use

TabPFN downloads its checkpoint the first time it runs, so **the board needs internet
for the first run**. If it does not have any, copy the checkpoint across from your
laptop instead:

```bash
laptop$ rsync -av ~/Library/Caches/tabpfn/ orangepi@<ip>:~/.cache/tabpfn/
```

Verify the destination path on the board afterwards; if the first run still tries to
download, let it, or check where it is looking with
`python -c "import tabpfn, pathlib; print(tabpfn.__file__)"`.

## 4. Run the benchmark — under tmux

```bash
pi$ tmux new -s bench
pi$ . .venv/bin/activate
pi$ make bench-hardware
```

`tmux` matters because an SSH drop kills a foreground job. Detach with `Ctrl-b d`,
reattach later with `tmux attach -t bench`.

This writes `results/hardware_bench.json`, stamped with the board string, CPU, RAM,
core count and thread setting — a latency figure without the machine and the thread
count is not reproducible, so the script refuses to produce one.

It reports three things:

- **Inference latency** in four configurations — cached and uncached KV, one and
  eight ensemble members — as seconds per batch and milliseconds per window.
- **The two design claims, measured on your board.** On the development laptop the
  KV cache is worth 7.3× and the 8-member ensemble costs 5.5×. If your board
  reproduces the *ordering*, the design claim holds; the magnitudes will differ.
- **A warning if there is no device tree**, because a laptop run of this script
  otherwise looks exactly like a board run.

Everything is single-threaded on purpose. torch and LightGBM each ship an OpenMP
runtime and crash the process together, and single-threaded is the honest gateway
configuration anyway — it costs about 2.5× against unpinned laptop numbers.

## 5. Get the results back

From your laptop:

```bash
laptop$ scp 'orangepi@<ip>:~/pump_monitoring/results/hardware_bench_*.json' \
        ~/pump_monitoring/results/
```

The filename already carries the board identity, so this cannot overwrite the laptop
run. Check the `platform.board` field inside before quoting anything from it: if it
reads `null` with `system: Darwin`, you are looking at a laptop file that reached the
board by rsync, not a measurement. Then copy the measured
latencies into §3.4 of the draft, replacing the laptop figures.

## 6. Optionally, re-run one experiment on the board

**This one does need the data** — about 115 MB of ESPset, which you would have to
rsync across (`--exclude data` above skipped it deliberately). Only do this if the
benchmark succeeded and you have time to spare.

```bash
laptop$ rsync -av ~/pump_monitoring/data/espset/ orangepi@<ip>:~/pump_monitoring/data/espset/
pi$     make experiment-espset   # not -full; that is ~40 min on a laptop
```

Only worth it if the install succeeded and you have time. It changes no scientific
conclusion — the same code produced the laptop numbers — but it lets you say the
cross-machine result was reproduced on the deployment hardware. If it is slow,
that *is* the finding, and it belongs in the paper.

---

## The Coral TPU: do not attempt it

You have one and the synopsis proposed it. It is the wrong accelerator for this
model, but be careful how you say so: **we never attempted the port**, so "it cannot
work" is not a claim this project has earned. What we have is a constraint analysis,
and that is still worth reporting.

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
restricted INT8 operator set; RKNN has the same requirement.

⚠️ **Detection is not the same question as compatibility.** `lsusb` enumerates at the
kernel level, before any driver is loaded, so an unsupported device still appears. If
the Coral does not show up there, that is a cable or power fault, not an
incompatibility — the two have different fixes and only one of them is interesting.

⚠️ **The shape argument alone is not decisive**, and a reviewer who has used these
tools will say so. Padding the reference set to a fixed maximum makes the graph
static — wasteful, since you pay attention over the padding on every query, but not
disqualifying. The two obstacles that padding does not fix are the real ones:

1. **The operator set** does not cover a transformer attention stack. Unsupported ops
   fall back to the CPU, so the model would largely run there anyway.
2. **INT8 quantisation of a prior-fitted model** without degrading the calibration
   that the abstention mechanism depends on is an open problem, not a build step.
3. **The userspace stack is older than the rest of the toolchain.** The runtime
   itself is fine — `libedgetpu1-std` installs for arm64 from Google's repo, though
   the available build is Debian bullseye running on Ubuntu jammy. The Python
   binding layer is the question, and this board runs Python 3.10. Verify rather
   than assume: `pip download pycoral`. A system meant to run unattended for years
   inherits the maintenance trajectory of every dependency it takes on — but do not
   write that the bindings *cannot* work unless you have tried and they did not.

So report it as a limitation of the deployment target, with the shape table as
context and those two as the substance — and say plainly that no port was attempted.
It is one honest paragraph. What it is **not** is a reason to abandon the
architecture: the gateway is a shared, mains-powered board where CPU inference is
affordable, and the accelerator was only ever an optimisation.

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

- [ ] `ssh` in, and `rsync` the ~11 MB tree across (no git remote exists yet)
- [ ] `cat /proc/device-tree/model` — know what you have
- [ ] `pip install -e ".[tabpfn]"` — time-boxed; abandon if torch fights you
- [ ] `make bench-hardware` **inside tmux** — the one command that matters
- [ ] `scp` the JSON back under a *different* filename from the laptop run
- [ ] Copy the measured numbers into §3.4, replacing the laptop figures
- [ ] Paste the shape table above into the Coral paragraph
- [ ] ❌ Do not attempt an Edge TPU or RKNN port
