.PHONY: install test report report-synthetic

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

report:
	python scripts/run_report.py

report-synthetic:
	python scripts/run_report.py --synthetic
