import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from signalforge.data import generate_synthetic_market_data
from signalforge.features.factory import FeatureFactory
from signalforge.validation.leakage import LeakageValidator


def test_leakage_validator_flags_negative_lag():
    validator = LeakageValidator()
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
        'future_return_5d': [0.1, 0.2, 0.3, 0.4, 0.5],
        'return_1d': [0.01, 0.02, 0.03, 0.04, 0.05],
        'sentiment_7d_mean': [0.2, 0.3, 0.4, 0.5, 0.6],
    })
    issues = validator.validate(df, {'return_1d': -1, 'sentiment_7d_mean': 7})
    assert any('negative lag' in issue for issue in issues)


def test_no_lookahead_in_features_truncation():
    """Verify features at time t do not change when future rows are truncated.

    Uses 300 days of sample data to ensure long-window features (sma_200, 52w high/low) are populated.
    """
    sample_df = generate_synthetic_market_data(days=300, symbols=("RELIANCE", "INFY"))
    factory = FeatureFactory()
    full = factory.generate(sample_df)

    cut_date = sorted(sample_df["timestamp"].unique())[-50]
    part_df = sample_df[sample_df["timestamp"] <= cut_date].copy()
    part = factory.generate(part_df)

    feature_cols = factory.get_feature_names(full)
    cols = [c for c in feature_cols if not c.startswith("future_")]

    key = ["symbol", "timestamp"]
    full_sub = full[full["timestamp"] <= cut_date].set_index(key)[cols].sort_index()
    part_sub = part.set_index(key)[cols].sort_index()

    # Evaluate non-NaN ratio on the post-warmup evaluation slice (after day 60)
    warmup_cutoff = sorted(sample_df["timestamp"].unique())[60]
    eval_slice = part_sub[part_sub.index.get_level_values("timestamp") >= warmup_cutoff]
    non_nan_ratio = float(eval_slice.notna().mean().mean())
    assert non_nan_ratio > 0.90, f"Vacuous truncation test: only {non_nan_ratio:.2%} non-NaN values on post-warmup slice"

    assert_frame_equal(full_sub, part_sub, check_exact=False, atol=1e-6)



def test_future_price_shuffling_does_not_affect_history():
    """Verify shuffling future prices does not alter historical feature values up to cut_date."""
    sample_df = generate_synthetic_market_data(days=60, symbols=("RELIANCE", "INFY"))
    factory = FeatureFactory()
    orig_features = factory.generate(sample_df)

    cut_date = sorted(sample_df["timestamp"].unique())[-20]

    shuffled_df = sample_df.copy()
    future_mask = shuffled_df["timestamp"] > cut_date
    shuffled_df.loc[future_mask, "close"] = np.random.permutation(shuffled_df.loc[future_mask, "close"].values)

    shuffled_features = factory.generate(shuffled_df)

    feature_cols = factory.get_feature_names(orig_features)
    cols = [c for c in feature_cols if not c.startswith("future_")]

    key = ["symbol", "timestamp"]
    orig_sub = orig_features[orig_features["timestamp"] <= cut_date].set_index(key)[cols].sort_index()
    shuf_sub = shuffled_features[shuffled_features["timestamp"] <= cut_date].set_index(key)[cols].sort_index()

    assert_frame_equal(orig_sub, shuf_sub, check_exact=False, atol=1e-6)
