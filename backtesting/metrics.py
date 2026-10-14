import numpy as np
import pandas as pd


def compute_ic(feature: pd.Series, target: pd.Series) -> float:
    x = feature.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = target.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    if x.size < 2 or y.size < 2:
        return 0.0
    corr = np.corrcoef(x, y)
    if corr.shape == (2, 2):
        return float(corr[0, 1])
    return 0.0


def compute_sharpe(daily_returns: pd.Series) -> float:
    returns = daily_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if returns.empty:
        return 0.0
    std = returns.std(ddof=1)
    if np.isclose(std, 0.0):
        return 0.0
    return float((returns.mean() / std) * np.sqrt(252))


def run_backtest(df: pd.DataFrame, prediction_col: str = "prediction", return_col: str = "future_return_5d") -> dict[str, float]:
    out = df.copy()
    out = out.dropna(subset=[prediction_col, return_col]).copy()
    out["strategy_return"] = out[prediction_col].clip(-1.0, 1.0) * out[return_col].clip(-1.0, 1.0)
    out = out.dropna(subset=["strategy_return"]).copy()
    cum = (1 + out["strategy_return"]).cumprod()
    sharpe = compute_sharpe(out["strategy_return"])
    metrics = {
        "sharpe": sharpe,
        "cumulative_return": float(cum.iloc[-1] - 1.0) if not cum.empty else 0.0,
        "mean_return": float(out["strategy_return"].mean()) if not out.empty else 0.0,
        "max_drawdown": float((cum / cum.cummax() - 1).min()) if not cum.empty else 0.0,
    }
    return metrics
