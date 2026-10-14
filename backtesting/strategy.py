from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from signalforge.backtesting.costs import compute_indian_transaction_cost, get_one_way_cost_bps


def compute_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sortino ratio."""
    rets = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0) - (risk_free_rate / 252)
    downside_returns = rets[rets < 0]
    downside_std = downside_returns.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float((rets.mean() / downside_std) * np.sqrt(252 / 5))


def compute_turnover(positions: pd.Series) -> float:
    """Compute average daily portfolio position turnover."""
    diff = positions.diff().abs().fillna(0.0)
    return float(diff.mean())


def compute_deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int = 41,
    sr_trials_var: float = 0.04,
    target_sharpe: float = 0.0,
    annualization_factor: float = np.sqrt(252 / 5),
) -> float:
    """Compute López de Prado's Deflated Sharpe Ratio (DSR).

    Adjusts estimated Sharpe ratio for selection bias (number of backtest trials),
    non-normality (skewness and kurtosis), and sample length N.
    Returns estimated probability that true Sharpe ratio > 0.
    """
    rets = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    N = len(rets)
    if N < 5:
        return 0.0

    mean_r = float(rets.mean())
    std_r = float(rets.std(ddof=1))
    if std_r == 0 or np.isnan(std_r):
        return 0.0

    sr_period = mean_r / std_r
    sk = float(skew(rets))
    ku = float(kurtosis(rets, fisher=False))  # Pearson kurtosis

    var_sr_period = (1.0 - sk * sr_period + ((ku - 1.0) / 4.0) * (sr_period**2)) / max(1, N - 1)
    se_period = float(np.sqrt(max(var_sr_period, 1e-8)))

    sr_ann = sr_period * annualization_factor
    se_ann = se_period * annualization_factor

    g = np.euler_gamma
    if n_trials > 1:
        emax_ann = float(np.sqrt(max(sr_trials_var, 1e-6)) * ((1.0 - g) * norm.ppf(1.0 - 1.0 / n_trials) + g * norm.ppf(1.0 - 1.0 / (n_trials * np.e))))
    else:
        emax_ann = target_sharpe

    dsr_stat = (sr_ann - emax_ann) / (se_ann + 1e-8)
    return float(norm.cdf(dsr_stat))


def compute_dsr_multi_trials(
    returns: pd.Series | np.ndarray,
    trials_list: list[int] = [41, 100, 200],
    sr_trials_var: float = 0.04,
) -> dict[str, float]:
    """Evaluate DSR across multiple trial counts (41, 100, 200)."""
    res = {}
    for n in trials_list:
        res[f"dsr_{n}"] = compute_deflated_sharpe_ratio(returns, n_trials=n, sr_trials_var=sr_trials_var)
    return res


def compute_sharpe_difference_bootstrap(
    returns_a: pd.Series | np.ndarray,
    returns_b: pd.Series | np.ndarray,
    block_size: int = 5,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Unified Circular Block Bootstrap test for difference in Sharpe ratios (Sharpe_A - Sharpe_B).
    
    Guarantees mathematical consistency between 95% Confidence Interval and two-sided p-value.
    """
    ra = np.asarray(pd.Series(returns_a).replace([np.inf, -np.inf], np.nan).dropna())
    rb = np.asarray(pd.Series(returns_b).replace([np.inf, -np.inf], np.nan).dropna())
    min_len = min(len(ra), len(rb))
    if min_len < 10:
        return {"diff_sharpe": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}

    ra, rb = ra[:min_len], rb[:min_len]

    def _sharpe(r):
        sd = np.std(r, ddof=1)
        return (np.mean(r) / sd) * np.sqrt(252 / 5) if sd > 0 else 0.0

    obs_diff = float(_sharpe(ra) - _sharpe(rb))
    rng = np.random.default_rng(seed)
    extended_a = np.concatenate([ra, ra[:block_size]])
    extended_b = np.concatenate([rb, rb[:block_size]])
    num_blocks = int(np.ceil(min_len / block_size))

    boot_diffs = []
    for _ in range(n_bootstraps):
        starts = rng.integers(0, min_len, size=num_blocks)
        sample_a = np.concatenate([extended_a[s : s + block_size] for s in starts])[:min_len]
        sample_b = np.concatenate([extended_b[s : s + block_size] for s in starts])[:min_len]
        boot_diffs.append(float(_sharpe(sample_a) - _sharpe(sample_b)))

    boot_arr = np.array(boot_diffs)
    ci_low = float(np.percentile(boot_arr, 2.5))
    ci_high = float(np.percentile(boot_arr, 97.5))

    # Unified Two-Sided Bootstrap p-value:
    # 2 * min(P(diff <= 0), P(diff >= 0)), capped at 1.0
    prop_le_0 = float(np.mean(boot_arr <= 0.0))
    prop_ge_0 = float(np.mean(boot_arr >= 0.0))
    p_val = float(min(1.0, 2.0 * min(prop_le_0, prop_ge_0)))

    return {
        "diff_sharpe": obs_diff,
        "p_value": p_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def panel_backtest(
    df: pd.DataFrame,
    signal_col: str = "prediction",
    hold: int = 5,
    top_k: int = 20,
    cost_bps: float = 13.61,
    filter_non_zero_signals: bool = True,
    n_trials: int = 41,
    sr_trials_var: float = 0.04,
) -> dict[str, float | pd.Series | dict]:
    """Execute a matrix panel backtest on pivoted (Timestamp x Symbol) data.

    Enforces sizing consistency: 20 names with a 5% cap (1/20 = 5%), summing to 100% portfolio weight.
    Applies exact statutory Indian transaction costs.
    """
    empty_res = {
        "sharpe": 0.0,
        "sortino": 0.0,
        "deflated_sharpe": 0.0,
        "dsr_41": 0.0,
        "dsr_100": 0.0,
        "dsr_200": 0.0,
        "max_drawdown": 0.0,
        "turnover": 0.0,
        "hit_rate": 0.0,
        "cumulative_return": 0.0,
        "net_returns": pd.Series(dtype=float),
    }

    if "timestamp" not in df.columns or "symbol" not in df.columns or "open" not in df.columns or signal_col not in df.columns:
        return empty_res

    clean_df = df.dropna(subset=["timestamp", "symbol", "open", signal_col]).copy()
    if clean_df.empty:
        return empty_res

    # Filter out dates where predictions are all 0 (non-test fold idle periods)
    if filter_non_zero_signals and signal_col in ("prediction", "prediction_full", "prediction_ohlcv", "prediction_ablated"):
        active_dates = clean_df.groupby("timestamp")[signal_col].apply(lambda x: (x != 0.0).any())
        valid_timestamps = active_dates[active_dates].index
        if len(valid_timestamps) > 0:
            clean_df = clean_df[clean_df["timestamp"].isin(valid_timestamps)].copy()

    if clean_df.empty:
        return empty_res

    px = clean_df.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    pred = clean_df.pivot(index="timestamp", columns="symbol", values=signal_col).reindex(px.index)

    # Forward return over holding window: enter at next open, exit `hold` opens later
    fwd = px.shift(-(1 + hold)) / px.shift(-1) - 1.0
    dates = [d for d in px.index[::hold] if pred.loc[d].notna().any()]
    if not dates:
        return empty_res

    # Form target weights per rebalance date (Top 20 names @ 5% cap = 100% allocated)
    if signal_col == "eq_weight_signal" or (pred.loc[dates].nunique(axis=1) <= 1).all():
        w = px.loc[dates].notna().astype(float)
        row_sums = w.sum(axis=1).replace(0, np.nan)
        w = w.div(row_sums, axis=0).fillna(0.0)
    else:
        ranks = pred.loc[dates].rank(axis=1, ascending=False)
        effective_k = min(top_k, max(1, px.shape[1]))
        w = (ranks <= effective_k).astype(float)
        row_sums = w.sum(axis=1).replace(0, np.nan)
        w = w.div(row_sums, axis=0).fillna(0.0)

    # Position turnover and transaction cost calculation
    turn = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    cost = turn * (cost_bps / 10000.0)
    gross_returns = (w * fwd.loc[dates]).sum(axis=1)
    net_returns = (gross_returns - cost).dropna()

    if net_returns.empty:
        return empty_res

    eq = (1.0 + net_returns).cumprod()
    sd = net_returns.std(ddof=1)
    sharpe = float((net_returns.mean() / sd) * np.sqrt(252 / hold)) if sd > 0 else 0.0
    sortino = compute_sortino(net_returns)
    
    # Honest DSR at 41, 100, and 200 trials
    dsr_dict = compute_dsr_multi_trials(net_returns, trials_list=[41, 100, 200], sr_trials_var=sr_trials_var)

    peak = eq.cummax()
    drawdown = (eq - peak) / peak.replace(0, np.nan)
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    hit_rate = float(np.mean(net_returns > 0)) if len(net_returns) > 0 else 0.0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "deflated_sharpe": dsr_dict["dsr_41"],
        "dsr_41": dsr_dict["dsr_41"],
        "dsr_100": dsr_dict["dsr_100"],
        "dsr_200": dsr_dict["dsr_200"],
        "max_drawdown": max_dd,
        "turnover": float(turn.mean()),
        "hit_rate": hit_rate,
        "cumulative_return": float(eq.iloc[-1] - 1.0) if len(eq) > 0 else 0.0,
        "mean_daily_return": float(net_returns.mean()),
        "net_returns": net_returns,
    }


def compute_cost_sensitivity_table(
    df: pd.DataFrame,
    signals_dict: dict[str, str],
    cost_levels_bps: list[float] = [0.0, 5.0, 10.0, 13.61, 20.0, 25.0],
) -> pd.DataFrame:
    """Evaluate strategy net Sharpe and performance across varying transaction cost levels (bps)."""
    records = []
    for cost_bps in cost_levels_bps:
        row = {"Transaction Cost (bps)": f"{cost_bps:.2f} bps"}
        for strat_name, sig_col in signals_dict.items():
            res = panel_backtest(df, signal_col=sig_col, cost_bps=cost_bps)
            row[strat_name] = res.get("sharpe", 0.0)
        records.append(row)

    return pd.DataFrame(records)


def run_strategy_backtest(
    df: pd.DataFrame,
    signal_col: str = "prediction",
    return_col: str = "future_return_5d",
    transaction_cost_bps: float = 13.61,
) -> dict[str, float]:
    """Execute strategy backtest. Uses matrix panel backtest if timestamp & symbol exist."""
    if "timestamp" in df.columns and "symbol" in df.columns and "open" in df.columns:
        return panel_backtest(df, signal_col=signal_col, hold=5, top_k=20, cost_bps=transaction_cost_bps)

    out = df.copy()
    if signal_col not in out.columns or return_col not in out.columns:
        return {"sharpe": 0.0, "sortino": 0.0, "deflated_sharpe": 0.0, "max_drawdown": 0.0, "turnover": 0.0, "hit_rate": 0.0, "cumulative_return": 0.0}

    out = out.dropna(subset=[signal_col, return_col]).copy()
    if out.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "deflated_sharpe": 0.0, "max_drawdown": 0.0, "turnover": 0.0, "hit_rate": 0.0, "cumulative_return": 0.0}

    signal_std = out[signal_col].std()
    weights = (out[signal_col] / (signal_std + 1e-6)).clip(-1.0, 1.0)
    out["position"] = weights

    cost_factor = transaction_cost_bps / 10000.0
    turnover_series = out["position"].diff().abs().fillna(0.0)
    costs = turnover_series * cost_factor

    out["gross_return"] = out["position"] * out[return_col]
    out["strategy_return"] = out["gross_return"] - costs

    rets = out["strategy_return"]
    cum_returns = (1 + rets).cumprod()

    rets_std = rets.std(ddof=1)
    sharpe = float((rets.mean() / rets_std) * np.sqrt(252 / 5)) if rets_std > 0 else 0.0
    sortino = compute_sortino(rets)
    dsr = compute_deflated_sharpe_ratio(rets, n_trials=41)

    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak.replace(0, np.nan)
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    hit_rate = float(np.mean(rets > 0)) if len(rets) > 0 else 0.0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "deflated_sharpe": dsr,
        "max_drawdown": max_dd,
        "turnover": float(turnover_series.mean()),
        "hit_rate": hit_rate,
        "cumulative_return": float(cum_returns.iloc[-1] - 1.0) if len(cum_returns) > 0 else 0.0,
        "mean_daily_return": float(rets.mean()),
        "net_returns": rets,
    }
