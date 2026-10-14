from __future__ import annotations

import pandas as pd


class LeakageValidator:
    """Rigorous leakage check for financial time-series features.

    Ensures feature publication/availability timestamp strictly precedes prediction target timestamp,
    detects target contamination, and flags invalid lags.
    """

    def __init__(self, timestamp_col: str = "timestamp", target_col: str = "future_return_5d"):
        self.timestamp_col = timestamp_col
        self.target_col = target_col

    def validate(self, df: pd.DataFrame, feature_lag_map: dict[str, int] | None = None) -> list[str]:
        """Return a list of leakage warnings."""
        issues = []
        feature_lag_map = feature_lag_map or {}

        # 1. Temporal monotonicity check
        if self.timestamp_col in df.columns:
            if not self.check_temporal_order(df):
                issues.append("Dataset timestamps are not strictly monotonically sorted per symbol.")

        # 2. Lag map verification
        for feature_name, lag in feature_lag_map.items():
            if feature_name not in df.columns:
                continue
            if lag < 0:
                issues.append(f"{feature_name}: negative lag ({lag}) indicates future look-ahead leakage")
            elif lag == 0 and "future" in feature_name:
                issues.append(f"{feature_name}: zero lag with target-like name may leak same-period target")

        # 3. Target contamination check
        if self.target_col in df.columns:
            future_columns = [c for c in df.columns if "future" in c.lower() and c != self.target_col]
            if future_columns:
                issues.append(f"Future target columns must be excluded from feature set: {future_columns}")

        return issues

    def check_temporal_order(self, df: pd.DataFrame) -> bool:
        if self.timestamp_col not in df.columns:
            return False
        if "symbol" in df.columns:
            for _, group in df.groupby("symbol"):
                if not group[self.timestamp_col].is_monotonic_increasing:
                    return False
            return True
        return df[self.timestamp_col].is_monotonic_increasing

    def verify_timestamp_availability(self, feature_timestamps: pd.Series, target_timestamps: pd.Series) -> bool:
        """Verify that every feature publication timestamp < target prediction timestamp."""
        return (feature_timestamps < target_timestamps).all()
