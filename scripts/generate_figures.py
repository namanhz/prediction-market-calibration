"""Generate all publication figures from pre-computed outputs.

Reads CSVs from output/{kalshi,bayesian,cross_platform}/ and saves
PNG + PDF to output/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUT_DIR
from src.plotting import (
    fig_hero_decomposition,
    fig_observed_vs_fitted,
    fig_slope_trajectories,
    fig_whale_effect,
)

KALSHI = OUTPUT_DIR / "kalshi"
FIGURES = OUTPUT_DIR / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 70)

    # Load decomposed calibration matrix
    cal_path = KALSHI / "calibration_matrix_decomposed.csv"
    if not cal_path.exists():
        cal_path = KALSHI / "calibration_matrix.csv"
    cal = pd.read_csv(cal_path)
    print(f"  Loaded {len(cal)} cells from {cal_path.name}")

    # Figure 1: Slope trajectories
    fig_slope_trajectories(cal, str(FIGURES / "figure1_slope_trajectories"))
    print("  saved figure1_slope_trajectories.{png,pdf}")

    # Figure 2: Hero decomposition (requires decomposition columns)
    if "mu" in cal.columns:
        fig_hero_decomposition(cal, str(FIGURES / "figure2_hero"))
        print("  saved figure2_hero.{png,pdf}")
    else:
        print("  SKIP figure2_hero (no decomposition columns — run run_kalshi.py first)")

    # Figure 3: Observed vs fitted
    if "fitted" in cal.columns:
        r2 = fig_observed_vs_fitted(cal, str(FIGURES / "figure3_observed_vs_fitted"))
        print(f"  saved figure3_observed_vs_fitted.{{png,pdf}}  R2={r2:.4f}")
    else:
        print("  SKIP figure3_observed_vs_fitted (no 'fitted' column)")

    # Figure 4: Whale effect
    fig_whale_effect(cal, str(FIGURES / "figure4_whale_effect"))
    print("  saved figure4_whale_effect.{png,pdf}")

    # Cross-platform figures (if available)
    cp_dir = OUTPUT_DIR / "cross_platform"
    if (cp_dir / "polymarket_slopes_by_domain_time.csv").exists():
        from src.plotting import (
            fig_cross_platform_trajectories,
            fig_politics_comparison,
            fig_scale_effect_comparison,
        )

        pm_dt = pd.read_csv(cp_dir / "polymarket_slopes_by_domain_time.csv")
        kalshi_dt = pd.read_csv(KALSHI / "calibration_slopes_by_domain_time.csv")
        fig_cross_platform_trajectories(pm_dt, kalshi_dt, str(FIGURES / "figure_cp1_slope_trajectories"))
        print("  saved figure_cp1_slope_trajectories.{png,pdf}")

        fig_politics_comparison(kalshi_dt, pm_dt, str(FIGURES / "figure_cp2_politics_comparison"))
        print("  saved figure_cp2_politics_comparison.{png,pdf}")

        if (cp_dir / "polymarket_slopes_by_domain_size.csv").exists():
            pm_ds = pd.read_csv(cp_dir / "polymarket_slopes_by_domain_size.csv")
            kalshi_ds = pd.read_csv(KALSHI / "calibration_slopes_by_domain_size.csv")
            fig_scale_effect_comparison(kalshi_ds, pm_ds, str(FIGURES / "figure_cp3_scale_effect"))
            print("  saved figure_cp3_scale_effect.{png,pdf}")
    else:
        print("  SKIP cross-platform figures (run run_cross_platform.py first)")

    print("\n" + "=" * 70)
    print(f"  DONE — figures in {FIGURES}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
