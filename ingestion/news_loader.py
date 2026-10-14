from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

from signalforge.features.indic_nlp import compute_indic_financial_sentiment, detect_indic_language

# Standard financial news headline templates for Indic multilingual news generation
NEWS_TEMPLATES = [
    {"lang": "hi", "text": "{symbol} का तिमाही परिणाम: मुनाफा 22% बढ़ा, लाभांश की घोषणा", "sentiment": 0.8},
    {"lang": "hi", "text": "{symbol} राजस्व और आय में जोरदार बढ़त, उत्पाद की बिक्री बढ़ी", "sentiment": 0.6},
    {"lang": "hi", "text": "{symbol} तिमाही नतीजों में घाटा, मंदी और गिरावट से दबाव", "sentiment": -0.7},
    {"lang": "hi", "text": "{symbol} मुकदमा और कर्ज जोखिम बढ़ा, शेयरों में भारी गिरावट", "sentiment": -0.8},
    {"lang": "en", "text": "{symbol} reports stellar Q2 earnings, revenue up 18% YoY", "sentiment": 0.75},
    {"lang": "en", "text": "{symbol} faces margin pressure amidst rising input costs", "sentiment": -0.5},
    {"lang": "hi", "text": "{symbol} नया ऑर्डर प्राप्त, उत्पादन और विकास में तेजी", "sentiment": 0.7},
    {"lang": "en", "text": "{symbol} regulatory inquiry raises compliance risk concern", "sentiment": -0.65},
]


def generate_indic_news_dataset(market_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate realistic timestamped Indic & English financial news dataset for NSE symbols."""
    if market_df.empty or "symbol" not in market_df.columns or "timestamp" not in market_df.columns:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    symbols = sorted(market_df["symbol"].unique())
    timestamps = sorted(market_df["timestamp"].unique())
    
    # Generate news events roughly every 5 trading days per symbol
    records = []
    for sym in symbols:
        # Pick dates periodically
        sample_dates = timestamps[::5]
        for ts in sample_dates:
            template = rng.choice(NEWS_TEMPLATES)
            headline = template["text"].format(symbol=sym)
            res = compute_indic_financial_sentiment(headline)
            lang = detect_indic_language(headline)
            
            records.append({
                "timestamp": pd.to_datetime(ts),
                "symbol": sym,
                "headline": headline,
                "news_sentiment": res["sentiment"],
                "indic_risk_signal": res["risk"],
                "positive_news": 1.0 if res["sentiment"] > 0 else 0.0,
                "negative_news": 1.0 if res["sentiment"] < 0 else 0.0,
                "risk_mentions": res["risk"],
                "revenue_mentions": res["revenue"],
                "management_confidence": max(0.0, res["sentiment"]),
                "language": lang,
            })

    out = pd.DataFrame(records)
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def load_news_data(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Load news CSV or JSON files and normalize into a standardized financial news dataframe."""
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            paths = sorted(p.rglob("*.csv")) + sorted(p.rglob("*.json"))
        else:
            paths = [p]

    frames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        if path.suffix == ".json":
            df = pd.read_json(path)
        else:
            df = pd.read_csv(path)

        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        rename_map = {
            "date": "timestamp",
            "datetime": "timestamp",
            "published_at": "timestamp",
            "ticker": "symbol",
            "company": "symbol",
            "text": "headline",
            "article": "headline",
            "sentiment": "news_sentiment",
        }
        df = df.rename(columns=rename_map)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper()

        if "news_sentiment" not in df.columns:
            df["news_sentiment"] = 0.0

        for col in ["positive_news", "negative_news", "risk_mentions", "revenue_mentions", "management_confidence"]:
            if col not in df.columns:
                df[col] = 0.0

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "news_sentiment", "headline", "language"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out
