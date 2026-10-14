from __future__ import annotations

import numpy as np
import pandas as pd


class QualityFilter:
    """Filter out bad quality features: constant, near-zero variance, excessive NaNs/Infs, or duplicates."""

    def __init__(self, max_missing_ratio: float = 0.25, min_variance: float = 1e-6):
        self.max_missing_ratio = max_missing_ratio
        self.min_variance = min_variance

    def filter_features(self, df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], dict[str, str]]:
        """Filter feature columns based on quality criteria.

        Returns (clean_feature_cols, rejection_reasons).
        """
        valid_cols = []
        rejections = {}

        for col in feature_cols:
            if col not in df.columns:
                continue

            series = df[col]
            missing_ratio = series.isna().mean()
            if missing_ratio > self.max_missing_ratio:
                rejections[col] = f"Excessive missing values ({missing_ratio:.1%})"
                continue

            clean_s = series.replace([np.inf, -np.inf], np.nan).dropna()
            if clean_s.nunique() <= 1:
                rejections[col] = "Constant feature (zero variance)"
                continue

            var = float(clean_s.var())
            if var < self.min_variance:
                rejections[col] = f"Near-constant feature (variance {var:.2e} < {self.min_variance:.2e})"
                continue

            valid_cols.append(col)

        # Duplicate checking
        final_cols = []
        seen_hashes = set()
        for col in valid_cols:
            s_clean = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            col_hash = hash(tuple(s_clean.values))
            if col_hash in seen_hashes:
                rejections[col] = "Duplicate feature vector"
                continue
            seen_hashes.add(col_hash)
            final_cols.append(col)

        return final_cols, rejections
