.PHONY: test figures experiment experiment-tabpfn demo-data lint \
        experiment-espset experiment-twente experiment-real experiment-espset-full \
        figures-espset figures-twente figures-summary figures-all rig-demo

PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)
export PYTHONPATH := src:$(PYTHONPATH)
# LightGBM and torch each ship an OpenMP runtime and segfault the process together
# on macOS. Held to one thread here as well as in run_experiment.py, so `make
# figures` and ad-hoc invocations get the same protection.
export OMP_NUM_THREADS := 1

test:
	$(PYTHON) -m pytest tests/ -q

figures:
	$(PYTHON) scripts/make_figures.py

demo-data:
	$(PYTHON) -c "from pathlib import Path; from pumpwatch.datasets.twente import write_demo_twente_cache; write_demo_twente_cache(Path('data/twente_demo'))"

experiment: demo-data
	$(PYTHON) scripts/run_experiment.py --skip-tabpfn

experiment-tabpfn: demo-data
	$(PYTHON) scripts/run_experiment.py

# --- real data -------------------------------------------------------------
# Both require a download first; each loader prints instructions if absent.
# ESPset: ~115 MB from doi.org/10.17632/m268jsw339.3 into data/espset/
# Twente: 20.8 GB from doi.org/10.4121/2b61183e-c14f-4131-829b-cc4822c369d0,
#         extracted (nested 7z) into data/raw/twente_sel/
experiment-espset:
	$(PYTHON) scripts/run_espset_experiment.py

experiment-twente:
	$(PYTHON) scripts/run_twente_experiment.py

experiment-real: experiment-espset experiment-twente

# Publication run: tuned baselines + multiple seeds. Slower; this is the one whose
# numbers are defensible, because it answers "you didn't tune the baseline" and
# "that's one seed" before a reviewer has to ask.
experiment-espset-full:
	$(PYTHON) scripts/run_espset_experiment.py --tune --seeds 5

# Figures, per dataset so the synthetic and real outputs cannot overwrite each other.
figures-espset:
	$(PYTHON) scripts/make_figures.py --results results/results_espset_both.json \
	  --outdir figures/espset

figures-twente:
	$(PYTHON) scripts/make_figures.py --results results/results_twente_real.json \
	  --outdir figures/twente

# Cross-dataset figures: leakage inflation across all three, and the PCA panels.
figures-summary:
	$(PYTHON) scripts/make_summary_figures.py

figures-all: figures figures-espset figures-twente figures-summary

# Runnable rig collection against the simulated backend, including the abort path.
rig-demo:
	$(PYTHON) scripts/collect_rig_data.py --root data/ownrig_demo \
	  --session-id demo_dry_1 --pump-id P1 --impeller-id I1 --bearing-id B1 \
	  --condition dry_run --severity 0.7 --rpm 1470 --ambient-temp 25 \
	  --duration-s 60 || true

lint:
	$(PYTHON) -m compileall -q src scripts tests
