from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from signalforge.modeling.sanitize import sanitize_frame

BASELINE_FEATURE_COLS = ["return_1d", "return_5d", "volume_change"]


class BaselineModel:
    """Baseline quantitative trading model using only price & volume returns."""

    def __init__(self, model=None):
        self.model = model or RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.feature_cols = BASELINE_FEATURE_COLS

    def fit(self, df: pd.DataFrame, target_col: str = "future_return_5d") -> BaselineModel:
        cols = [c for c in self.feature_cols if c in df.columns]
        X = sanitize_frame(df[cols], cols)
        y = sanitize_frame(df[[target_col]], [target_col])[target_col]
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        cols = [c for c in self.feature_cols if c in df.columns]
        X = sanitize_frame(df[cols], cols)
        return pd.Series(self.model.predict(X), index=df.index)
