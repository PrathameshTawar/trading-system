from __future__ import annotations

from typing import Any
import pandas as pd


def find_leads(df: pd.DataFrame, rules: dict[str, Any]) -> list[dict[str, Any]]:
    """Finds candidate volume & price momentum leads from market DataFrame.
    
    Deterministic code - zero LLM requirement.
    """
    leads: list[dict[str, Any]] = []
    watchlist = rules.get("watchlist", ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"])
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for sym in watchlist:
        sym_df = df[df["symbol"] == sym].sort_values("timestamp")
        if len(sym_df) < 10:
            continue

        d = sym_df.tail(31)
        curr_row = d.iloc[-1]
        prev_row = d.iloc[-2]

        vol_hist = d["volume"].iloc[:-1]
        mean_vol = vol_hist.mean() if not vol_hist.empty and vol_hist.mean() > 0 else 1.0
        curr_vol = float(curr_row["volume"])
        vol_ratio = curr_vol / mean_vol

        prev_close = float(prev_row["close"])
        curr_close = float(curr_row["close"])
        move = (curr_close / prev_close) - 1.0 if prev_close > 0 else 0.0

        # Trigger condition: Volume spike (>= 1.3x) or price move (>= 1.5%)
        if vol_ratio >= 1.3 or abs(move) >= 0.015:
            leads.append({
                "symbol": sym,
                "timestamp": str(curr_row["timestamp"].date() if hasattr(curr_row["timestamp"], "date") else curr_row["timestamp"]),
                "close": curr_close,
                "open": float(curr_row["open"]),
                "high": float(curr_row["high"]),
                "low": float(curr_row["low"]),
                "volume": curr_vol,
                "vol_ratio": round(vol_ratio, 2),
                "price_move_pct": round(move * 100, 2),
            })

    # Sort leads by highest volume ratio descending
    leads.sort(key=lambda x: x["vol_ratio"], reverse=True)
    return leads
