import numpy as np
import pandas as pd


def add_cross_asset_features(df: pd.DataFrame, benchmark_col: str = "nifty_close") -> pd.DataFrame:
    """Compute cross-asset, market relative strength, rolling beta, and market regime signals."""
    out = df.copy()
    if benchmark_col not in out.columns or "symbol" not in out.columns or "timestamp" not in out.columns:
        # Generate default synthetic benchmark columns if benchmark_col is missing
        out[benchmark_col] = out["close"].copy()

    base = out.sort_values(["symbol", "timestamp"]).copy()
    grouped = base.groupby("symbol", group_keys=False)

    bench_ret = grouped[benchmark_col].pct_change().fillna(0.0)
    stock_ret = grouped["close"].pct_change().fillna(0.0)

    # Relative returns vs market benchmark
    base["relative_return_vs_benchmark"] = (stock_ret - bench_ret).fillna(0.0)

    # Relative strength ratio
    base["market_relative_strength"] = (base["close"] / base[benchmark_col].replace(0, np.nan) - 1.0).fillna(0.0)
    base["sector_relative_strength"] = base["market_relative_strength"]

    # Rolling beta & rolling correlation over 20-day window
    def _rolling_beta_and_corr(g: pd.DataFrame) -> pd.DataFrame:
        s_ret = g["close"].pct_change().fillna(0.0)
        b_ret = g[benchmark_col].pct_change().fillna(0.0)

        cov = s_ret.rolling(20, min_periods=5).cov(b_ret)
        var = b_ret.rolling(20, min_periods=5).var()
        corr = s_ret.rolling(20, min_periods=5).corr(b_ret)

        beta = (cov / var.replace(0, np.nan)).fillna(1.0)
        return pd.DataFrame({"rolling_beta": beta, "correlation_to_market": corr.fillna(0.0)}, index=g.index)

    beta_corr_res = grouped.apply(_rolling_beta_and_corr, include_groups=False)
    if isinstance(beta_corr_res, pd.DataFrame) and "rolling_beta" in beta_corr_res.columns:
        if "symbol" in beta_corr_res.index.names:
            beta_corr_res = beta_corr_res.reset_index(level="symbol", drop=True)
        base["rolling_beta"] = beta_corr_res["rolling_beta"]
        base["correlation_to_market"] = beta_corr_res["correlation_to_market"]
    else:
        base["rolling_beta"] = 1.0
        base["correlation_to_market"] = 0.0

    base["beta_to_benchmark"] = base["rolling_beta"]

    # Market regime flag (1: Bullish, 0: Neutral, -1: Bearish)
    bench_ma50 = grouped[benchmark_col].transform(lambda s: s.rolling(50, min_periods=1).mean())
    base["market_regime"] = np.where(base[benchmark_col] > bench_ma50, 1.0, -1.0)

    return base
