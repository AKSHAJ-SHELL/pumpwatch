.PHONY: test figures experiment experiment-tabpfn demo-data lint

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

lint:
	$(PYTHON) -m compileall -q src scripts tests
