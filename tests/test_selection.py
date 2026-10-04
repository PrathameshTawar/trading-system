import pandas as pd

from signalforge.features.factory import FeatureFactory
from signalforge.selection.selector import FeatureSelector


def make_sample_df() -> pd.DataFrame:
    rows = []
    for day in range(60):
        for symbol in ['A', 'B']:
            close = 100 + day * (1 if symbol == 'A' else -1)
            sentiment = (day % 5) / 10
            rows.append({
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day),
                'symbol': symbol,
                'close': close,
                'volume': 1000 + day,
                'sentiment': sentiment,
            })
    return pd.DataFrame(rows)


def test_feature_selector_returns_candidates():
    df = FeatureFactory().generate(make_sample_df())
    selector = FeatureSelector(min_ic=0.0)
    selected = selector.select(df, target_col='future_return_5d')
    assert len(selected) >= 1
    assert 'return_1d' in selected or 'sentiment_3d_mean' in selected
