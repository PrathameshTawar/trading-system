import pandas as pd
from signalforge.backtesting.strategy import compute_sortino, compute_turnover, run_strategy_backtest


def test_strategy_backtest_metrics():
    df = pd.DataFrame({
        "prediction": [0.1, -0.2, 0.3, 0.05, -0.1, 0.4],
        "future_return_5d": [0.02, -0.01, 0.03, -0.005, -0.02, 0.04],
    })
    res = run_strategy_backtest(df, signal_col="prediction", return_col="future_return_5d")
    assert "sharpe" in res
    assert "sortino" in res
    assert "max_drawdown" in res
    assert "turnover" in res
    assert "hit_rate" in res
    assert res["hit_rate"] >= 0.0


def test_sharpe_difference_bootstrap_date_alignment():
    from signalforge.backtesting.strategy import compute_sharpe_difference_bootstrap
    # Create two return series with overlapping but staggered dates
    dates_b = pd.date_range("2022-01-01", periods=50, freq="5D")
    dates_a = dates_b[10:]  # Strategy A starts 10 periods later (e.g. test folds only)

    # Identical returns on overlapping dates
    sb = pd.Series(0.01, index=dates_b)
    sa = pd.Series(0.01, index=dates_a)

    boot = compute_sharpe_difference_bootstrap(sa, sb, n_bootstraps=100)
    # Since sa and sb are identical on their aligned dates, the observed difference must be 0.0
    assert abs(boot["diff_sharpe"]) < 1e-6, f"Aligned difference should be 0, got {boot['diff_sharpe']}"
    assert boot["ci_low"] <= 1e-6 and boot["ci_high"] >= -1e-6


def test_equal_weight_drift_turnover():
    from signalforge.backtesting.strategy import panel_backtest
    # Two stocks with fluctuating prices over 4 rebalance cycles
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df_list = []
    for d in dates:
        df_list.append({"timestamp": d, "symbol": "A", "open": 100.0 + (d.day % 5) * 5.0, "eq_weight_signal": 1.0})
        df_list.append({"timestamp": d, "symbol": "B", "open": 100.0 - (d.day % 5) * 4.0, "eq_weight_signal": 1.0})
    df = pd.DataFrame(df_list)

    res = panel_backtest(df, signal_col="eq_weight_signal", hold=5)
    # Drift-adjusted turnover should be non-zero due to relative price changes
    assert res["turnover"] > 0.0, f"Equal weight turnover should account for drift, got {res['turnover']}"
