import json
from pathlib import Path
import numpy as np
import pandas as pd

from signalforge.live.artifacts import load_bundle
from signalforge.live.paper import (
    compute_indian_transaction_cost,
    load_ledger,
    log_signals,
    paper_step,
    save_ledger,
)
from signalforge.live.positions import scores_to_cash_weights
from signalforge.live.score import score_universe, write_signals
from signalforge.live.train import train_production_model
from signalforge.live.validation import is_nse_trading_day, validate_daily_data
from signalforge.monitoring.drift import drift_status


def make_sample_df(days: int = 80, symbols=("AAA", "BBB", "CCC")) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        close = 100.0 + (hash(symbol) % 20)
        volume = 1000.0
        for day in range(days):
            close = close * (1 + 0.002 * ((day % 7) - 3) + 0.0004 * (1 if symbol == "AAA" else -1))
            volume = max(100.0, volume * (1 + 0.01 * ((day % 4) - 1.5)))
            open_px = close / 1.002
            rows.append(
                {
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "symbol": symbol,
                    "open": open_px,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": volume,
                    "sentiment": ((day % 7) - 3) / 10,
                    "positive_news": max(((day % 7) - 3) / 10, 0),
                    "negative_news": max(-((day % 7) - 3) / 10, 0),
                }
            )
    return pd.DataFrame(rows)


def test_cash_weights_are_long_only_and_capped():
    signals = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "prediction": [0.4, 0.2, -0.9, 0.05],
        }
    )
    out = scores_to_cash_weights(signals, long_only=True, max_names=2, max_weight=0.6, gross_exposure=1.0)
    assert out["weight"].sum() <= 1.0001
    assert (out["weight"] <= 0.6 + 1e-9).all()
    assert (out["weight"] > 0).sum() <= 2


def test_train_score_does_not_reselect(tmp_path: Path):
    df = make_sample_df()
    summary, model_dir = train_production_model(df, tmp_path / "models", selected_limit=8)
    frozen = list(summary["frozen_feature_cols"])
    bundle = load_bundle(model_dir)
    assert bundle.feature_cols == frozen
    assert (model_dir / "model.joblib").exists()

    signals, _drift, high_drift = score_universe(df, bundle)
    assert not signals.empty
    assert set(signals["symbol"]) == {"AAA", "BBB", "CCC"}
    assert "weight" in signals.columns
    assert high_drift in (True, False)
    path = write_signals(signals, tmp_path / "signals")
    assert path.exists()
    today = tmp_path / "signals" / "today.csv"
    assert today.exists()


def test_paper_fills_next_open(tmp_path: Path):
    df = make_sample_df(days=40)
    dates = sorted(pd.to_datetime(df["timestamp"]).dt.normalize().unique())
    d0, d1 = dates[-2], dates[-1]
    hist = df[pd.to_datetime(df["timestamp"]).dt.normalize() <= d0]
    _, model_dir = train_production_model(hist, tmp_path / "models", selected_limit=6)
    bundle = load_bundle(model_dir)

    signals, _, _ = score_universe(hist, bundle)
    ledger = load_ledger(tmp_path / "ledger.json", starting_cash=100_000.0, cost_bps=10.0)
    ledger = paper_step(hist, ledger, signals, signal_log_path=tmp_path / "signal_log.jsonl")
    assert ledger["pending"] is not None

    ledger = paper_step(df, ledger, None, signal_log_path=tmp_path / "signal_log.jsonl")
    assert ledger["pending"] is None
    save_ledger(ledger, tmp_path / "ledger.json")
    assert (tmp_path / "ledger.json").exists()
    assert "last_equity" in ledger
    assert ledger["last_equity"] > 0


def test_log_signals_idempotency_and_sha256(tmp_path: Path):
    log_file = tmp_path / "signal_log.jsonl"
    signals = pd.DataFrame(
        [
            {"symbol": "RELIANCE", "prediction": 0.0123, "weight": 0.05},
            {"symbol": "TCS", "prediction": -0.005, "weight": 0.0},
        ]
    )
    run_date = "2026-10-01"

    # First write should succeed
    res1 = log_signals(signals, run_date, path=log_file)
    assert res1 is True
    assert log_file.exists()

    # Second write for same date should return False (idempotent, no duplicate entry)
    res2 = log_signals(signals, run_date, path=log_file)
    assert res2 is False

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["date"] == run_date
    assert "sha256" in record
    assert len(record["sha256"]) == 64


def test_indian_fees_and_volume_slippage():
    cost_buy = compute_indian_transaction_cost(100_000.0, action="buy")
    cost_sell = compute_indian_transaction_cost(100_000.0, action="sell")
    assert cost_buy > 0 and cost_sell > 0
    # Buy includes stamp duty (0.015%), so buy cost > sell cost
    assert cost_buy > cost_sell


def test_nse_holiday_calendar():
    assert is_nse_trading_day("2026-10-01") is True  # Thursday
    assert is_nse_trading_day("2026-10-03") is False  # Saturday
    assert is_nse_trading_day("2026-10-02") is False  # Gandhi Jayanti (NSE Holiday)


def test_redesigned_drift_status_gate():
    ref_df = pd.DataFrame({"f1": np.random.normal(0, 1, 200), "f2": np.random.normal(0, 1, 200)})
    cur_small = pd.DataFrame({"f1": np.random.normal(0, 1, 20), "f2": np.random.normal(0, 1, 20)})
    # Too little data (< 100 rows) must return 'SKIP'
    assert drift_status(ref_df, cur_small, ["f1", "f2"], min_rows=100) == "SKIP"


def test_data_validation():
    df = make_sample_df(days=10)
    valid, rejections = validate_daily_data(df, expected_symbols=("AAA", "BBB", "CCC"))
    assert valid is True
    assert len(rejections) == 0


def test_shadow_portfolios_side_by_side(tmp_path: Path):
    from signalforge.live.paper import run_shadow_portfolios
    df = make_sample_df(days=50)
    _, model_dir = train_production_model(df, tmp_path / "models", selected_limit=6)
    bundle = load_bundle(model_dir)

    res = run_shadow_portfolios(df, bundle, base_dir=tmp_path / "shadow")
    assert "full_model" in res
    assert "equal_weight" in res
    assert "reversal_5d" in res
    assert "momentum_12_1" in res
    assert (tmp_path / "shadow" / "ledger_full_model.json").exists()
    assert (tmp_path / "shadow" / "signal_log_full_model.jsonl").exists()
