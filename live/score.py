from __future__ import annotations

from pathlib import Path

import pandas as pd

from signalforge.features.factory import FeatureFactory
from signalforge.live.artifacts import ProductionBundle
from signalforge.live.positions import scores_to_cash_weights
from signalforge.modeling.sanitize import sanitize_frame
from signalforge.live.alerts import send_alert
from signalforge.live.validation import validate_daily_data
from signalforge.monitoring.drift import compute_drift_report, drift_status


def latest_asof(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.Timestamp:
    return pd.to_datetime(df[timestamp_col]).max()


def score_universe(
    market_df: pd.DataFrame,
    bundle: ProductionBundle,
    factory: FeatureFactory | None = None,
    current_positions: set[str] | dict | None = None,
) -> tuple[pd.DataFrame, list[dict], bool]:
    """Score the latest complete bar per symbol with a frozen model. Does not re-select features."""
    valid, rejections = validate_daily_data(market_df)
    if not valid:
        send_alert("WARNING", "Data validation issues detected during universe scoring", {"reasons": rejections})

    factory = factory or FeatureFactory()
    features = factory.generate(market_df)
    asof = latest_asof(features, bundle.timestamp_col)
    latest = features[pd.to_datetime(features[bundle.timestamp_col]) == asof].copy()
    if latest.empty:
        raise ValueError("No rows available to score on the latest session.")

    missing = [c for c in bundle.feature_cols if c not in latest.columns]
    if missing:
        raise ValueError(f"Frozen features missing from live matrix: {missing}")

    X = sanitize_frame(latest[bundle.feature_cols], bundle.feature_cols)
    latest["prediction"] = bundle.model.predict(X)
    latest["signal_date"] = pd.to_datetime(asof).normalize()
    latest["execution"] = bundle.execution

    weighted = scores_to_cash_weights(
        latest,
        long_only=bundle.long_only,
        max_names=bundle.max_names,
        max_weight=bundle.max_weight,
        gross_exposure=bundle.gross_exposure,
        current_positions=current_positions,
    )

    drift_reports: list[dict] = []
    high_drift = False
    if not bundle.reference.empty:
        # Use recent multi-session evaluation window (rolling 20-60 sessions) for reliable PSI evaluation
        eval_window = features.sort_values(bundle.timestamp_col).tail(min(len(features), 3000))
        for feat in bundle.feature_cols[:8]:
            if feat in bundle.reference.columns and feat in eval_window.columns:
                rep = compute_drift_report(bundle.reference[feat], eval_window[feat], feature_name=feat)
                drift_reports.append(rep)

        d_status = drift_status(bundle.reference, eval_window, bundle.feature_cols, min_rows=100, psi_hi=0.25, min_breaches=3)
        if d_status == "HIGH":
            high_drift = True
            send_alert("HIGH", f"Redesigned drift gate triggered HIGH status on session {pd.to_datetime(asof).date()}", {"status": d_status})

    cols = [
        "signal_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "prediction",
        "weight",
        "execution",
    ]
    signals = weighted[[c for c in cols if c in weighted.columns]].sort_values("symbol")
    return signals, drift_reports, high_drift


def write_signals(signals: pd.DataFrame, output_dir: str | Path, asof: pd.Timestamp | None = None) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if asof is None:
        asof = pd.to_datetime(signals["signal_date"].iloc[0])
    path = out_dir / f"{pd.to_datetime(asof).strftime('%Y-%m-%d')}.csv"
    today = out_dir / "today.csv"
    signals.to_csv(path, index=False)
    signals.to_csv(today, index=False)
    return path
