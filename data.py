from __future__ import annotations

import pandas as pd


def generate_synthetic_market_data(days: int = 120, symbols: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    """Create a small synthetic market dataset for demo and testing use."""
    rows = []
    for symbol in symbols:
        close = 100.0
        volume = 1000
        for day in range(days):
            drift = 0.0008 * (1 if symbol == "A" else -0.5 if symbol == "B" else 0.2)
            shock = 0.002 * ((day % 7) - 3) / 3
            close = close * (1 + drift + shock)
            volume = int(volume * (1 + 0.02 * ((day % 5) - 2)))
            sentiment = ((day % 9) - 4) / 12

            rows.append(
                {
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "symbol": symbol,
                    "open": round(close * 0.995, 4),
                    "high": round(close * 1.02, 4),
                    "low": round(close * 0.98, 4),
                    "close": round(close, 4),
                    "volume": volume,
                    "sentiment": round(sentiment, 4),
                    "positive_news": max(sentiment, 0.0),
                    "negative_news": max(-sentiment, 0.0),
                }
            )
    return pd.DataFrame(rows)
