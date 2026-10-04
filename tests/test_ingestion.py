from pathlib import Path

import pandas as pd

from signalforge.features.cross_asset import add_cross_asset_features
from signalforge.ingestion.csv_loader import load_market_csvs, normalize_market_dataframe


def test_normalize_market_dataframe_handles_nse_bse_columns():
    df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Symbol': ['RELIANCE', 'RELIANCE'],
        'Open': [100.0, 101.0],
        'High': [101.5, 102.0],
        'Low': [99.5, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [5000, 6000],
    })
    normalized = normalize_market_dataframe(df, symbol_hint='RELIANCE')
    assert 'timestamp' in normalized.columns
    assert 'symbol' in normalized.columns
    assert 'close' in normalized.columns
    assert normalized['symbol'].nunique() == 1


def test_load_market_csvs_merges_multiple_files(tmp_path):
    file_a = tmp_path / 'RELIANCE.csv'
    file_b = tmp_path / 'TCS.csv'
    pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Open': [100, 101],
        'Close': [101, 102],
        'Volume': [1000, 1100],
    }).to_csv(file_a, index=False)
    pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Open': [200, 201],
        'Close': [201, 202],
        'Volume': [2000, 2100],
    }).to_csv(file_b, index=False)

    merged = load_market_csvs(tmp_path)
    assert set(merged['symbol']) == {'RELIANCE', 'TCS'}
    assert len(merged) == 4


def test_cross_asset_feature_generation_uses_benchmark_when_available():
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02', '2024-01-03']),
        'symbol': ['A', 'A', 'A', 'B', 'B', 'B'],
        'close': [100.0, 101.0, 103.0, 90.0, 89.0, 91.0],
        'nifty_close': [1000.0, 1005.0, 1015.0, 1000.0, 1005.0, 1015.0],
    })
    enriched = add_cross_asset_features(df)
    assert 'market_relative_strength' in enriched.columns
    assert 'relative_return_vs_benchmark' in enriched.columns
    assert 'beta_to_benchmark' in enriched.columns
    assert enriched['market_relative_strength'].notna().all()
