.PHONY: test figures experiment experiment-tabpfn demo-data lint bench-hardware \
        experiment-espset experiment-twente experiment-real experiment-espset-full \
        figures-espset figures-twente figures-summary figures-all rig-demo tables \
        gate-ablation experiment-paderborn operating-point

PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)
export PYTHONPATH := src:$(PYTHONPATH)
# LightGBM and torch each ship an OpenMP runtime and segfault the process together
# on macOS. Held to one thread here as well as in run_experiment.py, so `make
# figures` and ad-hoc invocations get the same protection.
export OMP_NUM_THREADS := 1

test:
	$(PYTHON) -m pytest tests/ -q

# Synthetic stand-in figures. Land in figures/synthetic/ so they cannot be mistaken
# for measurements; the real ones are figures/espset/ and figures/twente/.
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
	  --outdir figures/espset --no-shared-figures

figures-twente:
	$(PYTHON) scripts/make_figures.py --results results/results_twente_real.json \
	  --outdir figures/twente --no-shared-figures

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

# Isolates gate feature count from gate feature choice: every subset of the deployable
# features, at every size. Cheap - the gate is a Mahalanobis distance, not a model.
gate-ablation:
	$(PYTHON) scripts/gate_feature_ablation.py

# External check of the normalisation result on an independent dataset (Paderborn
# bearings, CC BY 4.0). Needs ~2 GB downloaded; the loader prints instructions.
experiment-paderborn:
	$(PYTHON) scripts/run_paderborn_experiment.py

# What would make the system deployable: sweeps decision cadence against the
# one-alarm-per-pump-per-month promise. The design's 5-minute cadence is a choice,
# not a constraint, and it is what caps end-to-end recall at 0.09.
operating-point:
	$(PYTHON) scripts/operating_point_study.py --model tabpfn_noabstain \
	  --out results/operating_point_study_tabpfn.json

# Results tables as markdown, read from results/*.json so the paper cannot cite a
# stale hand-copied number.
tables:
	$(PYTHON) scripts/make_paper_tables.py

# Run this ON the gateway board. Every latency number so far is laptop-measured,
# which is what makes "RK3588 gateway" a claim about unbenchmarked hardware.
bench-hardware:
	$(PYTHON) scripts/bench_gateway_hardware.py

lint:
	$(PYTHON) -m compileall -q src scripts tests
	# compileall catches syntax errors only; pyflakes catches the unused
	# imports and dead locals that accumulate through refactors.
	$(PYTHON) -m pyflakes src scripts
