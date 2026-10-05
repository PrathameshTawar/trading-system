from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_benchmark_data(filepath: str | Path, benchmark_name: str = "NIFTY") -> pd.DataFrame:
    """Load benchmark index series (e.g. NIFTY) and normalize columns for cross-asset merging."""
    path = Path(filepath)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    date_col = next((c for c in ["timestamp", "date", "trade_date"] if c in df.columns), None)
    close_col = next((c for c in ["close", "last", "index_value"] if c in df.columns), None)

    if not date_col or not close_col:
        return pd.DataFrame()

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[date_col]),
        f"{benchmark_name.lower()}_close": pd.to_numeric(df[close_col], errors="coerce"),
    })

    if "volume" in df.columns:
        out[f"{benchmark_name.lower()}_volume"] = pd.to_numeric(df["volume"], errors="coerce")

    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def merge_benchmark_with_market(market_df: pd.DataFrame, benchmark_df: pd.DataFrame, benchmark_col: str = "nifty_close") -> pd.DataFrame:
    """Merge benchmark index close prices into main market dataframe by timestamp."""
    if market_df.empty or benchmark_df.empty:
        return market_df

    out = market_df.copy()
    if "timestamp" not in out.columns or "timestamp" not in benchmark_df.columns:
        return out

    out["timestamp"] = pd.to_datetime(out["timestamp"])
    benchmark_df["timestamp"] = pd.to_datetime(benchmark_df["timestamp"])

    merged = out.merge(benchmark_df, on="timestamp", how="left")
    if benchmark_col in merged.columns:
        merged[benchmark_col] = merged[benchmark_col].ffill().bfill()

    return merged
