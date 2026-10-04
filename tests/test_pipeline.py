import pandas as pd

from signalforge.features.factory import FeatureFactory
from signalforge.pipelines.daily_pipeline import DailyPipeline


def make_sample_df(days: int = 40, symbols=('A', 'B')) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        close = 100.0
        volume = 1000
        for day in range(days):
            close = close * (1 + 0.001 * (day % 5 - 2) + 0.0005 * (1 if symbol == 'A' else -1))
            volume = volume * (1 + 0.01 * (day % 4 - 1.5))
            sentiment = ((day % 7) - 3) / 10
            rows.append({
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day),
                'symbol': symbol,
                'open': close,
                'high': close * 1.02,
                'low': close * 0.98,
                'close': close,
                'volume': volume,
                'sentiment': sentiment,
                'positive_news': max(sentiment, 0),
                'negative_news': max(-sentiment, 0),
            })
    return pd.DataFrame(rows)


def test_feature_factory_generates_features():
    df = make_sample_df()
    features = FeatureFactory().generate(df)
    assert 'return_1d' in features.columns
    assert 'future_return_5d' in features.columns
    assert 'sentiment_7d_mean' in features.columns
    assert not features.empty


def test_daily_pipeline_runs():
    pipeline = DailyPipeline()
    result = pipeline.run(make_sample_df())
    assert 'selected_features' in result
    assert 'walkforward' in result
    assert 'walk_forward_metrics' in result
    assert 'backtest_results' in result
    assert isinstance(result['selected_features'], list)
