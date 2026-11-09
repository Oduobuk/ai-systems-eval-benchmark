PYTHON ?= python3

.PHONY: help test demo clean

help:
	@echo "AI Systems Evaluation Benchmark - Command Menu:"
	@echo "  make test   - Run all schema and rubric scoring verification tests"
	@echo "  make demo   - Run demonstration benchmark across candidate model outputs"
	@echo "  make clean  - Remove build artifacts and caches"

test:
	$(PYTHON) -m pytest tests/ -v

demo:
	$(PYTHON) examples/run_mock_eval.py

clean:
	rm -rf build *.egg-info .pytest_cache eval_reports
	find . -type d -name __pycache__ -exec rm -rf {} +
