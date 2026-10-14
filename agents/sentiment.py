from __future__ import annotations

from typing import Any
import pandas as pd
from signalforge.features.indic_nlp import compute_indic_financial_sentiment


def evaluate_sentiment(df: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """Computes NLP and Indic news sentiment for target ticker."""
    sym_df = df[df["symbol"] == symbol] if not df.empty and "symbol" in df.columns else pd.DataFrame()
    if sym_df.empty or "sentiment" not in sym_df.columns:
        return {"symbol": symbol, "composite_sentiment": 0.0, "status": "NEUTRAL"}

    last_sentiment = float(sym_df["sentiment"].iloc[-1])
    status = "BULLISH" if last_sentiment > 0.1 else ("BEARISH" if last_sentiment < -0.1 else "NEUTRAL")

    return {
        "symbol": symbol,
        "composite_sentiment": round(last_sentiment, 4),
        "status": status,
    }
