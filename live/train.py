from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from signalforge.live.artifacts import ProductionBundle, save_bundle
from signalforge.modeling.sanitize import sanitize_frame
from signalforge.pipelines.daily_pipeline import DailyPipeline


def train_production_model(
    market_df: pd.DataFrame,
    model_dir: str | Path,
    *,
    selected_limit: int = 15,
    config_path: str | None = None,
    target_col: str = "future_return_5d",
    timestamp_col: str = "timestamp",
    long_only: bool = True,
    max_names: int = 5,
    max_weight: float = 0.30,
    gross_exposure: float = 1.0,
    cost_bps: float = 10.0,
) -> tuple[dict, Path]:
    """Run research pipeline once, then freeze selected features and a production model."""
    pipeline = DailyPipeline(
        target_col=target_col,
        timestamp_col=timestamp_col,
        config_path=config_path,
        selected_limit=selected_limit,
    )
    summary = pipeline.run(market_df)
    selected = list(summary.get("selected_features") or [])
    if not selected:
        raise RuntimeError("No features selected; refusing to freeze a live model.")

    features = pipeline.factory.generate(market_df)
    labeled = features.dropna(subset=[target_col]).copy()
    if labeled.empty:
        raise RuntimeError("No labeled rows (future_return_5d is all NaN). Need more history.")

    X = sanitize_frame(labeled[selected], selected)
    y = sanitize_frame(labeled[[target_col]], [target_col])[target_col]
    model = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42, n_jobs=1)
    model.fit(X, y)

    train_end = pd.to_datetime(labeled[timestamp_col]).max()
    sample_n = min(len(labeled), 5000)
    reference = labeled.sample(n=sample_n, random_state=42)[selected]

    bundle = ProductionBundle(
        model=model,
        feature_cols=selected,
        target_col=target_col,
        timestamp_col=timestamp_col,
        train_end=str(train_end),
        reference=reference,
        long_only=long_only,
        max_names=max_names,
        max_weight=max_weight,
        gross_exposure=gross_exposure,
        cost_bps=cost_bps,
        execution="next_open",
    )
    saved = save_bundle(bundle, model_dir)
    summary["frozen_model_dir"] = str(saved)
    summary["frozen_feature_cols"] = selected
    summary["train_end"] = str(train_end)
    summary["labeled_rows"] = int(len(labeled))
    return summary, saved
