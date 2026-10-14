from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


CANONICAL_COLUMNS = {
    "date": "timestamp",
    "datetime": "timestamp",
    "trade_date": "timestamp",
    "timestamp": "timestamp",
    "ticker": "symbol",
    "symbol": "symbol",
    "security": "symbol",
    "company": "symbol",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "last": "close",
    "prev_close": "prev_close",
    "volume": "volume",
    "turnover": "turnover",
    "sentiment": "sentiment",
    "news_sentiment": "sentiment",
    "positive_news": "positive_news",
    "negative_news": "negative_news",
}


def _infer_symbol_from_path(path: Path) -> str:
    return path.stem.upper().replace("-", "_").replace(" ", "_")


def normalize_market_dataframe(df: pd.DataFrame, symbol_hint: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    rename_map = {}
    for col in out.columns:
        canonical = CANONICAL_COLUMNS.get(col, col)
        if canonical != col:
            rename_map[col] = canonical
    out = out.rename(columns=rename_map)

    if "timestamp" not in out.columns and "date" not in out.columns:
        raise ValueError("CSV does not contain a date/timestamp column.")

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"])
    else:
        out["timestamp"] = pd.to_datetime(out["date"])

    for numeric_col in ["open", "high", "low", "close", "volume", "turnover", "prev_close", "sentiment", "positive_news", "negative_news"]:
        if numeric_col in out.columns:
            out[numeric_col] = pd.to_numeric(out[numeric_col], errors="coerce")

    if "symbol" not in out.columns:
        out["symbol"] = symbol_hint or _infer_symbol_from_path(Path("market.csv"))

    if "sentiment" not in out.columns:
        out["sentiment"] = 0.0
    if "positive_news" not in out.columns:
        out["positive_news"] = 0.0
    if "negative_news" not in out.columns:
        out["negative_news"] = 0.0

    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def load_market_csvs(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Load raw NSE/BSE CSVs and normalize them into a single DataFrame.

    This supports either a single file path or a directory containing CSV files.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    normalized_frames = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            csv_files = sorted(path.glob("*.csv"))
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                symbol_hint = csv_file.stem.upper().replace("-", "_")
                normalized_frames.append(normalize_market_dataframe(df, symbol_hint=symbol_hint))
        else:
            df = pd.read_csv(path)
            symbol_hint = path.stem.upper().replace("-", "_")
            normalized_frames.append(normalize_market_dataframe(df, symbol_hint=symbol_hint))

    if not normalized_frames:
        raise ValueError("No CSV files were found for ingestion.")

    return pd.concat(normalized_frames, ignore_index=True)
