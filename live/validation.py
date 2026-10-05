from __future__ import annotations

import pandas as pd
from signalforge.ingestion.market_data import DEFAULT_SYMBOLS

# Official NSE India Trading Holidays list for calendar filtering
NSE_HOLIDAYS = {
    "2021-01-26", "2021-03-29", "2021-04-02", "2021-04-14", "2021-04-21", "2021-05-13",
    "2021-07-21", "2021-08-19", "2021-09-10", "2021-10-15", "2021-11-04", "2021-11-05",
    "2021-11-19", "2022-01-26", "2022-03-01", "2022-03-18", "2022-04-14", "2022-04-15",
    "2022-05-03", "2022-08-09", "2022-08-15", "2022-08-31", "2022-10-05", "2022-10-24",
    "2022-10-26", "2022-11-08", "2023-01-26", "2023-03-07", "2023-03-30", "2023-04-04",
    "2023-04-07", "2023-04-14", "2023-05-01", "2023-06-29", "2023-08-15", "2023-09-19",
    "2023-10-02", "2023-10-24", "2023-11-14", "2023-11-27", "2023-12-25", "2024-01-22",
    "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29", "2024-04-11", "2024-04-17",
    "2024-05-01", "2024-05-20", "2024-06-17", "2024-07-17", "2024-08-15", "2024-10-02",
    "2024-11-01", "2024-11-15", "2024-12-25", "2025-01-26", "2025-02-26", "2025-03-14",
    "2025-03-31", "2025-04-10", "2025-04-14", "2025-04-18", "2025-05-01", "2025-08-15",
    "2025-08-27", "2025-10-02", "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    "2026-01-26", "2026-03-03", "2026-03-20", "2026-03-30", "2026-04-03", "2026-04-14",
    "2026-05-01", "2026-05-27", "2026-08-15", "2026-09-16", "2026-10-02", "2026-10-20",
    "2026-11-09", "2026-11-24", "2026-12-25"
}


def is_nse_trading_day(dt: str | pd.Timestamp) -> bool:
    """Check if a date is an active NSE trading day (excludes weekends and NSE holidays)."""
    ts = pd.to_datetime(dt)
    if ts.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    date_str = ts.strftime("%Y-%m-%d")
    return date_str not in NSE_HOLIDAYS


def validate_daily_data(
    market_df: pd.DataFrame,
    expected_symbols: tuple[str, ...] | None = None,
    max_price_gap_pct: float = 0.20,
) -> tuple[bool, list[str]]:
    """Validate daily market dataset before scoring or trading execution.
    
    Checks:
    - All expected symbols present in latest session
    - No missing/NaN/inf prices or volumes
    - No zero-volume rows
    - Flags price gaps > 20% (corporate actions, splits, or data corruptions)
    """
    reasons: list[str] = []
    if market_df.empty:
        return False, ["Market DataFrame is empty."]

    market_df = market_df.copy()
    market_df["timestamp"] = pd.to_datetime(market_df["timestamp"])
    asof = market_df["timestamp"].max()

    today_bars = market_df[market_df["timestamp"] == asof].copy()
    present_symbols = set(today_bars["symbol"].unique())

    if expected_symbols is not None and set(expected_symbols).intersection(present_symbols):
        expected_set = set(expected_symbols)
    else:
        expected_set = present_symbols

    missing_syms = expected_set - present_symbols
    min_threshold = min(len(expected_set), 10) if len(expected_set) >= 10 else len(expected_set)
    if len(present_symbols) < min_threshold:
        reasons.append(f"Insufficient universe size on {asof.date()}: found {len(present_symbols)} symbols, missing {len(missing_syms)} expected symbols.")

    # 1. Null / Inf value checks
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col in today_bars.columns:
            null_count = today_bars[col].isna().sum()
            if null_count > 0:
                reasons.append(f"Column '{col}' has {null_count} null/NaN values on {asof.date()}.")

    # 2. Zero volume check
    if "volume" in today_bars.columns:
        zero_vol_count = (today_bars["volume"] <= 0).sum()
        if zero_vol_count > 0:
            reasons.append(f"Found {zero_vol_count} symbols with zero or negative volume on {asof.date()}.")

    # 3. Single-day price jump / corporate action check (> 20% jump)
    if "close" in market_df.columns:
        grouped = market_df.sort_values(["symbol", "timestamp"]).groupby("symbol")
        recent_returns = grouped["close"].pct_change().abs()
        today_returns = recent_returns[market_df["timestamp"] == asof]
        large_gaps = today_returns[today_returns > max_price_gap_pct]
        if not large_gaps.empty:
            for idx in large_gaps.index:
                sym = market_df.loc[idx, "symbol"]
                gap_val = market_df.loc[idx, "close"]
                reasons.append(f"Large price gap (> {max_price_gap_pct*100:.0f}%) detected for {sym} on {asof.date()}: {today_returns.loc[idx]*100:.1f}%. Possible corporate action or split.")

    is_valid = len(reasons) == 0
    return is_valid, reasons
