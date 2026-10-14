from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

from signalforge.modeling.sanitize import sanitize_frame
from signalforge.validation.quality import QualityFilter


class FeatureSelector:
    """Multi-stage intelligence feature selector.

    Ranks candidate features using:
    - Information Coefficient (Mean IC, Median IC, IC Std, ICIR, % Positive IC)
    - Mutual Information
    - Pairwise Correlation Redundancy
    - Tree Model Feature Importance
    - Temporal Stability across time periods
    - Composite Feature Score
    """

    def __init__(
        self,
        min_ic: float = 0.01,
        max_corr: float = 0.90,
        max_missing_ratio: float = 0.25,
        weights: dict[str, float] | None = None,
    ):
        self.min_ic = min_ic
        self.max_corr = max_corr
        self.quality_filter = QualityFilter(max_missing_ratio=max_missing_ratio)
        self.weights = weights or {
            "mean_ic": 0.35,
            "icir": 0.25,
            "stability": 0.20,
            "importance": 0.20,
        }

    @staticmethod
    def compute_ic_stats(feature: pd.Series, target: pd.Series, group_col: pd.Series | None = None) -> dict[str, float]:
        """Compute Information Coefficient metrics: Mean IC, Median IC, IC Std, ICIR, % Positive IC."""
        valid_mask = feature.notna() & target.notna() & ~feature.isin([np.inf, -np.inf]) & ~target.isin([np.inf, -np.inf])
        if not valid_mask.any():
            return {"mean_ic": 0.0, "median_ic": 0.0, "ic_std": 0.0, "icir": 0.0, "pct_positive_ic": 0.0}

        x = feature[valid_mask]
        y = target[valid_mask]

        if x.nunique() <= 1 or y.nunique() <= 1:
            return {"mean_ic": 0.0, "median_ic": 0.0, "ic_std": 0.0, "icir": 0.0, "pct_positive_ic": 0.0}

        if group_col is not None:
            g = group_col[valid_mask]
            unique_g = g.unique()
            if len(unique_g) > 1:
                if len(unique_g) > 100:
                    rng = np.random.default_rng(42)
                    sample_g = set(rng.choice(unique_g, size=100, replace=False))
                    sub_m = g.isin(sample_g)
                    x, y, g = x[sub_m], y[sub_m], g[sub_m]

                tmp = pd.DataFrame({"x": x, "y": y, "g": g})
                tmp["rx"] = tmp.groupby("g")["x"].rank()
                tmp["ry"] = tmp.groupby("g")["y"].rank()
                
                # Vectorized Spearman rank correlation per timestamp group
                cov_series = tmp.groupby("g", observed=True).apply(
                    lambda sub: sub["rx"].corr(sub["ry"]) if len(sub) > 2 and sub["rx"].std() > 0 and sub["ry"].std() > 0 else np.nan
                )
                ic_list = cov_series.dropna().tolist()

                if ic_list:
                    ic_arr = np.array(ic_list)
                    mean_ic = float(np.mean(ic_arr))
                    std_ic = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 1e-4
                    icir = float(mean_ic / (std_ic + 1e-6))

                    from signalforge.modeling.walk_forward import compute_ic_inference
                    inf = compute_ic_inference(ic_arr)

                    return {
                        "mean_ic": mean_ic,
                        "median_ic": float(np.median(ic_arr)),
                        "ic_std": std_ic,
                        "icir": icir,
                        "nw_se": inf["nw_se"],
                        "ic_tstat": inf["t_stat"],
                        "ci_low": inf["ci_low"],
                        "ci_high": inf["ci_high"],
                        "pct_positive_ic": float(np.mean(ic_arr > 0)),
                    }

        # Overall correlation fallback
        corr, _ = spearmanr(x.to_numpy(), y.to_numpy())
        ic_val = 0.0 if np.isnan(corr) else float(corr)
        return {
            "mean_ic": ic_val,
            "median_ic": ic_val,
            "ic_std": 0.0,
            "icir": abs(ic_val),
            "nw_se": 0.0,
            "ic_tstat": 0.0,
            "ci_low": ic_val,
            "ci_high": ic_val,
            "pct_positive_ic": 1.0 if ic_val > 0 else 0.0,
        }

    def compute_stability(self, df: pd.DataFrame, feature: str, target_col: str, date_col: str = "timestamp", n_splits: int = 4) -> float:
        """Compute IC temporal stability across n_splits chronological periods."""
        if date_col not in df.columns:
            return 1.0

        sorted_df = df.sort_values(date_col)
        dates = sorted_df[date_col].drop_duplicates().tolist()
        if len(dates) < n_splits * 2:
            return 1.0

        split_dates = np.array_split(dates, n_splits)
        fold_ics = []
        for split in split_dates:
            sub = sorted_df[sorted_df[date_col].isin(split)]
            if len(sub) > 5:
                ic_stats = self.compute_ic_stats(sub[feature], sub[target_col])
                fold_ics.append(ic_stats["mean_ic"])

        if not fold_ics:
            return 0.0

        mean_abs_ic = float(np.mean(np.abs(fold_ics)))
        std_ic = float(np.std(fold_ics))
        if mean_abs_ic == 0:
            return 0.0
        stability = max(0.0, 1.0 - (std_ic / (mean_abs_ic + 1e-6)))
        return float(min(1.0, stability))

    def evaluate_features(self, df: pd.DataFrame, target_col: str = "future_return_5d", feature_cols: list[str] | None = None) -> pd.DataFrame:
        """Evaluate and rank all candidate features returning full evaluation table."""
        if feature_cols is None:
            exclude = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "turnover", target_col}
            feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("future_")]

        # Stage 1: Quality filtering
        clean_cols, rejections = self.quality_filter.filter_features(df, feature_cols)
        if not clean_cols:
            return pd.DataFrame()

        # Stage 2: IC calculation & Mutual Information
        X_clean = sanitize_frame(df[clean_cols], clean_cols)
        y_clean = sanitize_frame(df[[target_col]], [target_col])[target_col]

        try:
            if len(X_clean) > 5000:
                rng = np.random.default_rng(42)
                sub_idx = rng.choice(len(X_clean), size=5000, replace=False)
                X_mi, y_mi = X_clean.iloc[sub_idx], y_clean.iloc[sub_idx]
            else:
                X_mi, y_mi = X_clean, y_clean
            mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
            mi_map = dict(zip(clean_cols, mi_scores))
        except Exception:
            mi_map = {c: 0.0 for c in clean_cols}


        # Stage 3: Model importance
        try:
            if len(X_clean) > 5000:
                rng = np.random.default_rng(42)
                sub_idx = rng.choice(len(X_clean), size=5000, replace=False)
                X_rf, y_rf = X_clean.iloc[sub_idx], y_clean.iloc[sub_idx]
            else:
                X_rf, y_rf = X_clean, y_clean
            model = RandomForestRegressor(n_estimators=20, max_depth=6, random_state=42, n_jobs=1)
            model.fit(X_rf, y_rf)
            imp_map = dict(zip(clean_cols, model.feature_importances_))
        except Exception:
            imp_map = {c: 0.0 for c in clean_cols}

        records = []
        for col in clean_cols:
            group_col = df["timestamp"] if "timestamp" in df.columns else None
            ic_stats = self.compute_ic_stats(df[col], df[target_col], group_col=group_col)
            stability = self.compute_stability(df, col, target_col)

            mean_ic = ic_stats["mean_ic"]
            icir = ic_stats["icir"]
            ic_tstat = ic_stats.get("ic_tstat", 0.0)
            mi = mi_map.get(col, 0.0)
            imp = imp_map.get(col, 0.0)

            # Composite Score
            score = (
                self.weights["mean_ic"] * abs(mean_ic)
                + self.weights["icir"] * min(abs(icir), 3.0) / 3.0
                + self.weights["stability"] * stability
                + self.weights["importance"] * min(imp * 10, 1.0)
            )

            records.append({
                "feature": col,
                "mean_ic": mean_ic,
                "median_ic": ic_stats["median_ic"],
                "ic_std": ic_stats["ic_std"],
                "icir": icir,
                "nw_se": ic_stats.get("nw_se", 0.0),
                "ic_tstat": ic_tstat,
                "ci_low": ic_stats.get("ci_low", mean_ic),
                "ci_high": ic_stats.get("ci_high", mean_ic),
                "pct_positive_ic": ic_stats["pct_positive_ic"],
                "mutual_info": mi,
                "stability": stability,
                "model_importance": imp,
                "score": score,
            })

        res_df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)
        return res_df

    def select(self, df: pd.DataFrame, target_col: str = "future_return_5d", feature_cols: list[str] | None = None, top_k: int | None = None) -> list[str]:
        """Perform multi-stage IC, quality, redundancy, stability, and model selection.

        Returns ranked list of selected feature names.
        """
        eval_df = self.evaluate_features(df, target_col=target_col, feature_cols=feature_cols)
        if eval_df.empty:
            return []

        # Filter by minimum IC threshold
        candidates = eval_df[eval_df["mean_ic"].abs() >= self.min_ic]["feature"].tolist()
        if not candidates:
            candidates = eval_df["feature"].head(10).tolist()

        # Redundancy filtering (pairwise correlation pruning)
        selected = []
        for feat in candidates:
            if not selected:
                selected.append(feat)
                continue

            # Compute correlation with already selected features
            feat_s = df[feat].fillna(0.0).to_numpy()
            is_redundant = False
            for existing in selected:
                exist_s = df[existing].fillna(0.0).to_numpy()
                if np.std(feat_s) == 0 or np.std(exist_s) == 0:
                    corr = 0.0
                else:
                    corr = np.corrcoef(feat_s, exist_s)[0, 1]
                if not np.isnan(corr) and abs(corr) > self.max_corr:
                    is_redundant = True
                    break

            if not is_redundant:
                selected.append(feat)

        if top_k:
            selected = selected[:top_k]

        return selected
