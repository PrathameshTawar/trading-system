from __future__ import annotations

from pathlib import Path

import pandas as pd

from signalforge.ingestion.csv_loader import normalize_market_dataframe


DEFAULT_SYMBOLS = (
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "GODREJCP", "KOTAKBANK", "LT", "AXISBANK", "HCLTECH", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "ULTRACEMCO", "ICICIPRULI", "NTPC",
    "ONGC", "POWERGRID", "ADANIENT", "ADANIPORTS", "COALINDIA", "TATASTEEL", "JSWSTEEL",
    "M&M", "GRASIM", "HEROMOTOCO", "BAJAJ-AUTO", "EICHERMOT", "BPCL", "CIPLA",
    "DRREDDY", "DIVISLAB", "APOLLOHOSP", "TATACONSUM", "BRITANNIA", "PIDILITIND", "HDFCLIFE",
    "SBILIFE", "WIPRO", "TECHM", "HINDALCO", "NESTLEIND", "SHRIRAMFIN", "BEL"
)





def download_nse_daily(
    symbols: list[str] | tuple[str, ...] = DEFAULT_SYMBOLS,
    period: str = "5y",
) -> pd.DataFrame:
    """Download 5-year NSE cash equity daily bars and real ^NSEI benchmark data."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for live EOD download. Install with: pip install yfinance"
        ) from exc

    import time

    # Download real ^NSEI benchmark index data first
    bench_raw = None
    for attempt in range(3):
        try:
            bench_raw = yf.download("^NSEI", period=period, interval="1d", progress=False, auto_adjust=False)
            if bench_raw is not None and not bench_raw.empty:
                break
        except Exception:
            time.sleep(1.0)

    bench_df = None
    if bench_raw is not None and not bench_raw.empty:
        bench_df = bench_raw.reset_index()
        if isinstance(bench_df.columns, pd.MultiIndex):
            bench_df.columns = [col[0] for col in bench_df.columns]
        bench_df["timestamp"] = pd.to_datetime(bench_df["Date"])
        bench_df = bench_df[["timestamp", "Close"]].rename(columns={"Close": "nifty_close"})

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for sym in symbols:
        ticker = f"{sym}.NS" if not str(sym).startswith("^") else str(sym)
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                time.sleep(1.0)

        if raw is None or raw.empty:
            failures.append(sym)
            continue
        df = raw.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df["Symbol"] = sym
        norm_df = normalize_market_dataframe(df, symbol_hint=sym)
        
        if bench_df is not None:
            norm_df["timestamp"] = pd.to_datetime(norm_df["timestamp"])
            norm_df = pd.merge(norm_df, bench_df, on="timestamp", how="left")
            norm_df["nifty_close"] = norm_df["nifty_close"].ffill().bfill()

        frames.append(norm_df)


    if not frames:
        raise RuntimeError(
            "No NSE daily bars downloaded for "
            f"{list(symbols)}. Failures: {failures or 'unknown'}. "
            "Synthetic fallback is disabled for live-market use."
        )

    if failures:
        raise RuntimeError(
            f"Incomplete universe: missing {failures}. Refusing to train/trade on a partial book."
        )

    out = pd.concat(frames, ignore_index=True)

    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(f"Downloaded data missing columns: {sorted(missing)}")
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def save_market_csv(df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output
