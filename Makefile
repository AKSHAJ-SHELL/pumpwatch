.PHONY: test figures experiment experiment-tabpfn demo-data lint

PYTHON ?= python
export PYTHONPATH := src:$(PYTHONPATH)

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
