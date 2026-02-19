"""All robustness checks: clustered bootstrap, price range, weighted decomposition, ANOVA, confound.

Outputs to output/robustness/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import (
    compute_weighted_decomposition,
    decompose,
    fit_logistic,
    fit_slope,
    fit_slope_with_se,
)
from src.classify import get_group
from src.config import (
    BIN_LABELS,
    CELL_MIN,
    DATE_CUTOFF,
    DOMAINS,
    KALSHI_MARKETS,
    KALSHI_TRADES,
    OUTPUT_DIR,
    SIZE_LABELS,
    TIME_BINS,
)
from src.pipeline import size_bin_sql, time_bin_sql

OUT = OUTPUT_DIR / "robustness"
OUT.mkdir(parents=True, exist_ok=True)
KALSHI_OUT = OUTPUT_DIR / "kalshi"


# ═══════════════════════════════════════════════════════════════════
# FIX 1: Market-clustered bootstrap for whale effect
# ═══════════════════════════════════════════════════════════════════

def fix1_clustered_bootstrap():
    """Market-clustered bootstrap for the whale effect."""
    print("\n" + "=" * 70)
    print("  FIX 1: MARKET-CLUSTERED BOOTSTRAP")
    print("=" * 70)

    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")
    conn = duckdb.connect()

    df = conn.execute(f"""
        WITH resolved AS (
            SELECT ticker, event_ticker, result, close_time
            FROM '{markets}/*.parquet'
            WHERE status='finalized' AND result IN ('yes','no')
        ),
        trade_data AS (
            SELECT t.yes_price, t.count AS weight,
                   CASE WHEN m.result='yes' THEN 1 ELSE 0 END AS is_yes,
                   CASE WHEN t.count = 1 THEN 0 WHEN t.count <= 10 THEN 1
                        WHEN t.count <= 100 THEN 2 ELSE 3 END AS sbin,
                   m.ticker,
                   regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix
            FROM '{trades}/*.parquet' t
            INNER JOIN resolved m ON t.ticker = m.ticker
            WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
              AND m.close_time > t.created_time
              AND t.yes_price BETWEEN 5 AND 95
        ),
        market_counts AS (
            SELECT ticker FROM trade_data GROUP BY ticker HAVING COUNT(*) >= 10
        )
        SELECT td.ticker, td.sbin, td.yes_price, td.is_yes,
               SUM(td.weight) AS weight, td.cat_prefix
        FROM trade_data td
        INNER JOIN market_counts mc ON td.ticker = mc.ticker
        GROUP BY td.ticker, td.sbin, td.yes_price, td.is_yes, td.cat_prefix
    """).df()
    conn.close()

    df["domain"] = df["cat_prefix"].apply(get_group)
    print(f"  Loaded {len(df):,} aggregated rows")

    N_ITER = 5000
    np.random.seed(42)
    prices_flat = np.repeat(np.arange(5, 96), 2).astype(float)
    outcomes_flat = np.tile(np.array([0.0, 1.0]), 91)

    results_rows = []
    for domain in ["Politics", "Sports"]:
        print(f"\n  {domain}:")
        dom_df = df[df["domain"] == domain].copy()
        tickers = dom_df["ticker"].unique()
        n_tickers = len(tickers)
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        weight_tensor = np.zeros((n_tickers, 91, 2, 4), dtype=np.float64)
        ti = dom_df["ticker"].map(ticker_to_idx).values
        pi = dom_df["yes_price"].values.astype(int) - 5
        oi = dom_df["is_yes"].values.astype(int)
        si = dom_df["sbin"].values.astype(int)
        wv = dom_df["weight"].values.astype(float)
        np.add.at(weight_tensor, (ti, pi, oi, si), wv)

        print(f"    {n_tickers} markets")
        diffs = np.full(N_ITER, np.nan)
        for i in range(N_ITER):
            idx = np.random.randint(0, n_tickers, size=n_tickers)
            counts = np.bincount(idx, minlength=n_tickers).astype(np.float64)
            total_w = np.einsum("i,ijkl->jkl", counts, weight_tensor)

            w_large = total_w[:, :, 3].ravel()
            w_single = total_w[:, :, 0].ravel()
            mask_l = w_large > 0
            mask_s = w_single > 0

            sl = fit_slope(prices_flat[mask_l], outcomes_flat[mask_l], w_large[mask_l]) if mask_l.sum() >= 5 else np.nan
            ss = fit_slope(prices_flat[mask_s], outcomes_flat[mask_s], w_single[mask_s]) if mask_s.sum() >= 5 else np.nan
            diffs[i] = sl - ss

        valid = diffs[~np.isnan(diffs)]
        ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
        mean_diff = np.mean(valid)
        print(f"    Market-clustered: {mean_diff:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

        results_rows.append(dict(
            domain=domain, method="market_clustered",
            mean_diff=round(mean_diff, 4), ci_lo=round(ci_lo, 4),
            ci_hi=round(ci_hi, 4), n_valid=len(valid),
        ))

    result = pd.DataFrame(results_rows)
    result.to_csv(OUT / "whale_effect_clustered_bootstrap.csv", index=False)
    print("\n  saved whale_effect_clustered_bootstrap.csv")


# ═══════════════════════════════════════════════════════════════════
# FIX 5: Extended price range robustness
# ═══════════════════════════════════════════════════════════════════

def fix5_price_range():
    """Test decomposition stability across different price ranges."""
    print("\n" + "=" * 70)
    print("  FIX 5: PRICE RANGE ROBUSTNESS")
    print("=" * 70)

    markets = str(KALSHI_MARKETS).replace("\\", "/")
    trades = str(KALSHI_TRADES).replace("\\", "/")
    tb = time_bin_sql()
    sb = size_bin_sql()

    def load_and_fit(price_lo, price_hi):
        conn = duckdb.connect()
        raw = conn.execute(f"""
            WITH resolved AS (
                SELECT ticker, event_ticker, result, close_time
                FROM '{markets}/*.parquet'
                WHERE status='finalized' AND result IN ('yes','no')
            ),
            trade_data AS (
                SELECT t.yes_price, t.count AS trade_count,
                       CASE WHEN m.result='yes' THEN 1 ELSE 0 END AS is_yes,
                       regexp_extract(m.event_ticker, '^([A-Z0-9]+)', 1) AS cat_prefix,
                       EXTRACT(EPOCH FROM (m.close_time - t.created_time))/3600.0 AS hours_to_close,
                       ({sb}) AS sbin, m.ticker
                FROM '{trades}/*.parquet' t
                INNER JOIN resolved m ON t.ticker = m.ticker
                WHERE t.created_time <= TIMESTAMP '{DATE_CUTOFF}'
                  AND m.close_time > t.created_time
            ),
            market_counts AS (
                SELECT ticker FROM trade_data GROUP BY ticker HAVING COUNT(*) >= 10
            )
            SELECT td.cat_prefix, ({tb}) AS tbin, td.sbin, td.yes_price, td.is_yes,
                   SUM(td.trade_count) AS total_contracts, COUNT(*) AS n_trades
            FROM trade_data td
            INNER JOIN market_counts mc ON td.ticker = mc.ticker
            WHERE td.yes_price BETWEEN {price_lo} AND {price_hi} AND ({tb}) >= 0 AND td.sbin >= 0
            GROUP BY td.cat_prefix, ({tb}), td.sbin, td.yes_price, td.is_yes
        """).df()
        conn.close()

        raw["domain"] = raw["cat_prefix"].apply(get_group)
        raw = raw[raw["domain"].isin(DOMAINS)].copy()

        rows = []
        for (domain, tbin, sbin), cell in raw.groupby(["domain", "tbin", "sbin"]):
            n_t = int(cell["n_trades"].sum())
            if n_t < CELL_MIN:
                continue
            result = fit_slope_with_se(
                cell["yes_price"].values.astype(float),
                cell["is_yes"].values.astype(float),
                cell["total_contracts"].values.astype(float),
            )
            if result is None:
                continue
            b, a, se = result
            rows.append(dict(
                domain=domain, time_bin=BIN_LABELS[int(tbin)],
                size_bin=SIZE_LABELS[int(sbin)],
                n_trades=n_t, slope_b=b, slope_stderr=se,
            ))
        return pd.DataFrame(rows)

    configs = [(5, 95, "Baseline [5,95]"), (2, 98, "Extended [2,98]"), (1, 99, "Full [1,99]")]
    decomp_rows = []
    for lo, hi, label in configs:
        print(f"\n  Computing {label}...")
        cal = load_and_fit(lo, hi)
        cal = decompose(cal.copy())
        theta = cal["slope_b"].values
        ss_total = np.sum((theta - theta.mean()) ** 2)
        fitted_cumul = np.zeros(len(theta))
        prev = 0.0
        for comp in ["mu", "alpha", "beta", "gamma"]:
            fitted_cumul = fitted_cumul + cal[comp].values
            c_r2 = np.sum((fitted_cumul - theta.mean()) ** 2) / ss_total if ss_total > 0 else 0
            m_r2 = c_r2 - prev
            decomp_rows.append(dict(check_name=label, component=comp,
                                    marginal_r2=round(m_r2, 4), cumulative_r2=round(c_r2, 4)))
            prev = c_r2
        print(f"    {len(cal)} cells, total R²={prev:.4f}")

    pd.DataFrame(decomp_rows).to_csv(OUT / "price_range_robustness.csv", index=False)
    print("  saved price_range_robustness.csv")


# ═══════════════════════════════════════════════════════════════════
# FIX 2: Weighted variance decomposition
# ═══════════════════════════════════════════════════════════════════

def fix2_weighted_decomposition():
    """Compare unweighted vs inverse-variance weighted decomposition."""
    print("\n" + "=" * 70)
    print("  FIX 2: WEIGHTED DECOMPOSITION")
    print("=" * 70)

    cal = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv")
    slopes = cal["slope_b"].values
    time_bins = cal["time_bin"].values
    domains_arr = cal["domain"].values
    size_bins = cal["size_bin"].values
    se = cal["slope_stderr"].values

    uw = compute_weighted_decomposition(slopes, time_bins, domains_arr, size_bins, np.ones(len(slopes)))
    weights = 1.0 / (se ** 2)
    wt = compute_weighted_decomposition(slopes, time_bins, domains_arr, size_bins, weights)

    print(f"\n  {'Component':<12} {'Unweighted':>12} {'Weighted':>12}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12}")
    for comp in ["mu", "alpha", "beta", "gamma"]:
        print(f"  {comp:<12} {uw[comp]:>12.4f} {wt[comp]:>12.4f}")
    print(f"  {'TOTAL':<12} {uw['total']:>12.4f} {wt['total']:>12.4f}")

    rows = []
    for comp in ["mu", "alpha", "beta", "gamma"]:
        rows.append({"type": "Unweighted", "component": comp, "marginal_r2": round(uw[comp], 4)})
        rows.append({"type": "Weighted", "component": comp, "marginal_r2": round(wt[comp], 4)})
    pd.DataFrame(rows).to_csv(OUT / "weighted_variance_decomposition.csv", index=False)
    print("  saved weighted_variance_decomposition.csv")


# ═══════════════════════════════════════════════════════════════════
# FIX 3: Type I/II/III ANOVA
# ═══════════════════════════════════════════════════════════════════

def fix3_anova():
    """Compare Type I, II, III sums of squares."""
    print("\n" + "=" * 70)
    print("  FIX 3: TYPE I/II/III ANOVA")
    print("=" * 70)

    cal = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv")
    y = cal["slope_b"].values
    n = len(y)

    def one_hot(values):
        levels = sorted(set(values))
        mat = np.zeros((len(values), len(levels) - 1))
        for j, level in enumerate(levels[1:]):
            mat[:, j] = (np.array(values) == level).astype(float)
        return mat

    def interaction(v1, v2):
        oh1, oh2 = one_hot(v1), one_hot(v2)
        cols = []
        for i in range(oh1.shape[1]):
            for j in range(oh2.shape[1]):
                cols.append(oh1[:, i] * oh2[:, j])
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    X_time = one_hot(cal["time_bin"].values)
    X_domain = one_hot(cal["domain"].values)
    X_td = interaction(cal["time_bin"].values, cal["domain"].values)
    X_ds = interaction(cal["domain"].values, cal["size_bin"].values)
    intercept = np.ones((n, 1))

    def ss_res(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(np.sum((y - X @ beta) ** 2))

    models = {
        "null": intercept,
        "T": np.hstack([intercept, X_time]),
        "T+D": np.hstack([intercept, X_time, X_domain]),
        "T+D+TD": np.hstack([intercept, X_time, X_domain, X_td]),
        "full": np.hstack([intercept, X_time, X_domain, X_td, X_ds]),
    }
    ss = {k: ss_res(v) for k, v in models.items()}
    ss_total = float(np.sum((y - y.mean()) ** 2))

    type1 = {
        "time_bin": ss["null"] - ss["T"],
        "domain": ss["T"] - ss["T+D"],
        "time:domain": ss["T+D"] - ss["T+D+TD"],
        "domain:size": ss["T+D+TD"] - ss["full"],
    }

    models_drop = {
        "time_bin": np.hstack([intercept, X_domain, X_td, X_ds]),
        "domain": np.hstack([intercept, X_time, X_td, X_ds]),
        "time:domain": np.hstack([intercept, X_time, X_domain, X_ds]),
        "domain:size": np.hstack([intercept, X_time, X_domain, X_td]),
    }
    type3 = {k: ss_res(v) - ss["full"] for k, v in models_drop.items()}

    models_type2 = {
        "time_bin": (np.hstack([intercept, X_domain, X_ds]),
                     np.hstack([intercept, X_time, X_domain, X_ds])),
        "domain": (np.hstack([intercept, X_time, X_ds]),
                   np.hstack([intercept, X_time, X_domain, X_ds])),
        "time:domain": (np.hstack([intercept, X_time, X_domain, X_ds]),
                        np.hstack([intercept, X_time, X_domain, X_td, X_ds])),
        "domain:size": (np.hstack([intercept, X_time, X_domain, X_td]),
                        np.hstack([intercept, X_time, X_domain, X_td, X_ds])),
    }
    type2 = {k: ss_res(v[0]) - ss_res(v[1]) for k, v in models_type2.items()}

    rows = []
    for term in ["time_bin", "domain", "time:domain", "domain:size"]:
        rows.append(dict(
            term=term,
            type_I_SS=round(type1[term], 6), type_I_pct=round(100 * type1[term] / ss_total, 2),
            type_II_SS=round(type2[term], 6), type_II_pct=round(100 * type2[term] / ss_total, 2),
            type_III_SS=round(type3[term], 6), type_III_pct=round(100 * type3[term] / ss_total, 2),
        ))

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "anova_type_comparison.csv", index=False)
    print("  saved anova_type_comparison.csv")
    print(f"\n  {'Term':<15} {'Type I %':>10} {'Type II %':>10} {'Type III %':>10}")
    for _, r in result.iterrows():
        print(f"  {r['term']:<15} {r['type_I_pct']:>10.2f} {r['type_II_pct']:>10.2f} {r['type_III_pct']:>10.2f}")


# ═══════════════════════════════════════════════════════════════════
# FIX 4: Size x Horizon confound
# ═══════════════════════════════════════════════════════════════════

def fix4_size_horizon_confound():
    """Check if size effect persists after controlling for horizon."""
    print("\n" + "=" * 70)
    print("  FIX 4: SIZE x HORIZON CONFOUND")
    print("=" * 70)

    cal = pd.read_csv(KALSHI_OUT / "calibration_matrix.csv")
    y = cal["slope_b"].values
    n = len(y)

    def one_hot(values):
        levels = sorted(set(values))
        mat = np.zeros((len(values), len(levels) - 1))
        for j, level in enumerate(levels[1:]):
            mat[:, j] = (np.array(values) == level).astype(float)
        return mat

    def interaction(v1, v2):
        oh1, oh2 = one_hot(v1), one_hot(v2)
        cols = []
        for i in range(oh1.shape[1]):
            for j in range(oh2.shape[1]):
                cols.append(oh1[:, i] * oh2[:, j])
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    intercept = np.ones((n, 1))
    X_time = one_hot(cal["time_bin"].values)
    X_domain = one_hot(cal["domain"].values)
    X_td = interaction(cal["time_bin"].values, cal["domain"].values)
    X_ds = interaction(cal["domain"].values, cal["size_bin"].values)
    X_ts = interaction(cal["time_bin"].values, cal["size_bin"].values)

    def ss_res(X):
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(np.sum((y - X @ beta) ** 2))

    ss_total = float(np.sum((y - y.mean()) ** 2))

    ss_base = ss_res(np.hstack([intercept, X_time, X_domain, X_td, X_ds]))
    ss_ext = ss_res(np.hstack([intercept, X_time, X_domain, X_td, X_ds, X_ts]))
    ss_no_ds = ss_res(np.hstack([intercept, X_time, X_domain, X_td]))
    ss_with_ts = ss_res(np.hstack([intercept, X_time, X_domain, X_td, X_ts]))

    gamma_without_ts = (ss_no_ds - ss_base) / ss_total
    gamma_with_ts = (ss_with_ts - ss_ext) / ss_total
    ts_marginal = (ss_base - ss_ext) / ss_total

    print(f"  gamma_d(s) without tau x s: {gamma_without_ts:.4f}")
    print(f"  gamma_d(s) with tau x s:    {gamma_with_ts:.4f}")
    print(f"  tau x s marginal R2:        {ts_marginal:.4f}")

    # Within-time-bin whale effect for Politics
    pol = cal[cal["domain"] == "Politics"]
    whale_rows = []
    for tb in BIN_LABELS:
        single = pol[(pol["time_bin"] == tb) & (pol["size_bin"] == "Single")]
        large = pol[(pol["time_bin"] == tb) & (pol["size_bin"] == "Large")]
        s_val = float(single["slope_b"].iloc[0]) if len(single) > 0 else np.nan
        l_val = float(large["slope_b"].iloc[0]) if len(large) > 0 else np.nan
        whale_rows.append(dict(time_bin=tb, single_slope=round(s_val, 4),
                               large_slope=round(l_val, 4), difference=round(l_val - s_val, 4)))

    all_rows = [
        dict(metric="gamma_without_ts", value=round(gamma_without_ts, 4)),
        dict(metric="gamma_with_ts", value=round(gamma_with_ts, 4)),
        dict(metric="ts_marginal", value=round(ts_marginal, 4)),
    ]
    pd.DataFrame(all_rows).to_csv(OUT / "size_horizon_confound.csv", index=False)
    pd.DataFrame(whale_rows).to_csv(OUT / "politics_whale_by_time.csv", index=False)
    print("  saved size_horizon_confound.csv, politics_whale_by_time.csv")


def main():
    print("=" * 70)
    print("  ROBUSTNESS CHECKS")
    print("=" * 70)

    fix1_clustered_bootstrap()
    fix5_price_range()
    fix2_weighted_decomposition()
    fix3_anova()
    fix4_size_horizon_confound()

    print("\n" + "=" * 70)
    print(f"  DONE — outputs in {OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
