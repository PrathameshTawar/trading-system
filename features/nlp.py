import numpy as np
import pandas as pd
from signalforge.features.indic_nlp import add_indic_nlp_features


def add_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute financial NLP signals across multiple lookback windows and transformations."""
    out = df.copy()
    grouped = out.groupby("symbol", group_keys=False)

    sentiment_col = "news_sentiment" if "news_sentiment" in out.columns else "sentiment"
    if sentiment_col not in out.columns:
        out[sentiment_col] = 0.0

    # Multi-window rolling means
    out["sentiment_1d"] = grouped[sentiment_col].shift(1).fillna(0.0)
    out["sentiment_3d_mean"] = grouped[sentiment_col].transform(lambda s: s.rolling(3, min_periods=1).mean()).fillna(0.0)
    out["sentiment_7d_mean"] = grouped[sentiment_col].transform(lambda s: s.rolling(7, min_periods=1).mean()).fillna(0.0)
    out["sentiment_30d_mean"] = grouped[sentiment_col].transform(lambda s: s.rolling(30, min_periods=1).mean()).fillna(0.0)

    # Sentiment velocity & acceleration
    out["sentiment_change"] = grouped[sentiment_col].diff().fillna(0.0)
    out["sentiment_acceleration"] = grouped["sentiment_change"].diff().fillna(0.0)

    # Normalized z-score over 30d window
    mean_30d = grouped[sentiment_col].transform(lambda s: s.rolling(30, min_periods=1).mean())
    std_30d = grouped[sentiment_col].transform(lambda s: s.rolling(30, min_periods=1).std()).replace(0, np.nan)
    out["sentiment_zscore"] = ((out[sentiment_col] - mean_30d) / std_30d).fillna(0.0)

    # Positive and negative news ratios
    if "positive_news" in out.columns:
        out["positive_news_ratio"] = grouped["positive_news"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    else:
        out["positive_news_ratio"] = (out[sentiment_col] > 0).astype(float)

    if "negative_news" in out.columns:
        out["negative_news_ratio"] = grouped["negative_news"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    else:
        out["negative_news_ratio"] = (out[sentiment_col] < 0).astype(float)

    # Domain specific event flags
    if "risk_mentions" not in out.columns:
        out["risk_mentions"] = grouped[sentiment_col].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: float((x < -0.2).mean())))
    if "revenue_mentions" not in out.columns:
        out["revenue_mentions"] = grouped[sentiment_col].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: float((x > 0.15).mean())))
    if "management_confidence" not in out.columns:
        out["management_confidence"] = grouped[sentiment_col].transform(lambda s: s.rolling(5, min_periods=1).mean())

    # Add Indic NLP features if text is present
    out = add_indic_nlp_features(out)
    return out
