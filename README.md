# Prediction Market Calibration: A Microstructure Analysis

Replication code for the paper submitted to the *Journal of the Royal Statistical Society: Series A*.

## Overview

This repository contains all code to reproduce the empirical analysis of prediction market calibration across Kalshi and Polymarket. The analysis fits logistic recalibration models, performs ANOVA-style variance decomposition, estimates Bayesian hierarchical models, and runs robustness checks.

## Repository Structure

```
src/
  config.py           Constants, paths, bin definitions
  classify.py          Domain classification (Kalshi taxonomy + Polymarket regex)
  calibration.py       Logistic recalibration, decomposition, bootstrap
  bayesian.py          NumPyro hierarchical models (M0, M1, M2)
  pipeline.py          SQL helpers, DuckDB data loading
  plotting.py          matplotlib figure generation

scripts/
  run_kalshi.py            Main Kalshi analysis pipeline
  run_bayesian.py          Bayesian hierarchical models + LOO-CV
  run_cross_platform.py    Polymarket replication
  run_robustness.py        All robustness checks
  generate_figures.py      Publication figures

tests/
  test_imports.py          Smoke tests
```

## Data Requirements

This repository does not include data. The raw Parquet data (~36 GiB) is available from the companion data repository:

```bash
# Clone the data repo and download data
git clone https://github.com/Jon-Becker/prediction-market-analysis.git
cd prediction-market-analysis
make setup
```

By default, scripts look for data at `../prediction-market-analysis/data/`. Override with:

```bash
export DATA_DIR=/path/to/data
```

## Reproduction

### Quick Start

```bash
# Install dependencies
pip install -e .

# Run all analyses
make reproduce
```

### Step by Step

```bash
# 1. Kalshi calibration (requires data)
make kalshi

# 2. Bayesian models (requires calibration data CSV)
make bayesian

# 3. Cross-platform comparison (requires Polymarket unified data)
make cross-platform

# 4. Robustness checks (requires Kalshi data + step 1 outputs)
make robustness

# 5. Generate all publication figures
make figures
```

### Tests

```bash
make test
```

## Key Results

- **87.3%** of calibration variance explained by 4-component decomposition
- Universal horizon effect: 30.2% | Domain intercepts: 14.6% | Domain x horizon: 26.0% | Scale effects: 16.5%
- Politics domain intercept: alpha = +0.151 (Bayesian), +0.156 (frequentist)
- Politics whale effect: Delta = +0.53 [0.29, 0.75] on Kalshi; +0.11 [-0.15, 0.39] on Polymarket (not significant)
- Bayesian PPC coverage: 96.3% (208/216 cells), max R-hat = 1.000, min ESS = 4,070
- Cross-platform: Political underconfidence replicates; scale effect does not

## License

MIT License. See [LICENSE](LICENSE).
