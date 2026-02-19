.PHONY: reproduce kalshi bayesian cross-platform robustness figures test clean lint format

reproduce: kalshi bayesian cross-platform robustness figures

kalshi:
	python scripts/run_kalshi.py

bayesian:
	python scripts/run_bayesian.py

cross-platform:
	python scripts/run_cross_platform.py

robustness:
	python scripts/run_robustness.py

figures:
	python scripts/generate_figures.py

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

clean:
	rm -rf output/
