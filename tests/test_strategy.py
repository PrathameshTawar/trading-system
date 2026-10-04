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
