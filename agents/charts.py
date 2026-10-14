from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def levels(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """Computes technical price levels, ATR stop distance, and support/resistance for a given ticker."""
    sym_df = df[df["symbol"] == symbol].copy()
    if sym_df.empty:
        return {
            "symbol": symbol,
            "price": 0.0,
            "entry": 0.0,
            "stop": 0.0,
            "target": 0.0,
            "atr_14": 0.0,
            "data_age_days": 999,
        }

    sym_df["timestamp"] = pd.to_datetime(sym_df["timestamp"])
    sym_df = sym_df.sort_values("timestamp")

    curr_row = sym_df.iloc[-1]
    last_dt = pd.to_datetime(curr_row["timestamp"]).tz_localize(None)
    now_dt = pd.Timestamp.now().tz_localize(None)
    data_age_days = (now_dt - last_dt).days

    price = float(curr_row["close"])
    entry = price

    # Compute 14-period ATR
    high = sym_df["high"]
    low = sym_df["low"]
    close = sym_df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14, min_periods=1).mean().iloc[-1])

    # Stop Loss & Target calculation
    stop_distance = max(atr_14 * 1.5, price * 0.015)
    stop = max(1.0, price - stop_distance)
    target = price + (stop_distance * 2.0)  # 2.0x Risk-Reward ratio

    # 52-week High/Low & SMA-200
    tail_252 = sym_df.tail(252)
    high_52w = float(tail_252["high"].max())
    low_52w = float(tail_252["low"].min())
    sma_200 = float(sym_df["close"].rolling(200, min_periods=1).mean().iloc[-1])

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "atr_14": round(atr_14, 2),
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "sma_200": round(sma_200, 2),
        "dist_from_52w_high_pct": round(((price / high_52w) - 1.0) * 100, 2) if high_52w > 0 else 0.0,
        "data_age_days": data_age_days,
        "last_date": str(last_dt.date()),
    }
