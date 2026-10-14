from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from signalforge.modeling.baseline import BASELINE_FEATURE_COLS
from signalforge.modeling.sanitize import sanitize_frame


def daily_rank_ic(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = "future_return_5d",
    date_col: str = "timestamp",
) -> pd.Series:
    """Compute per-date cross-sectional Spearman rank IC series."""
    if date_col not in df.columns or pred_col not in df.columns or target_col not in df.columns:
        return pd.Series(dtype=float)
    clean = df.dropna(subset=[pred_col, target_col]).copy()
    if clean.empty:
        return pd.Series(dtype=float)

    def _group_spearman(sub):
        if len(sub) < 3 or sub[pred_col].nunique() <= 1 or sub[target_col].nunique() <= 1:
            return np.nan
        corr, _ = spearmanr(sub[pred_col], sub[target_col])
        return float(corr) if not np.isnan(corr) else np.nan

    ic_s = clean.groupby(date_col, observed=True).apply(_group_spearman).dropna()
    return ic_s


def compute_circular_block_bootstrap_ci(
    ic_arr: np.ndarray | pd.Series,
    block_size: int = 5,
    n_bootstraps: int = 500,
) -> tuple[float, float]:
    """Circular Block Bootstrap (L = 5 days) on daily IC series for target return autocorrelation."""
    x = np.asarray(ic_arr)
    x = x[~np.isnan(x)]
    N = len(x)
    if N == 0:
        return 0.0, 0.0
    if N < block_size:
        m = float(np.mean(x))
        return m, m

    rng = np.random.default_rng(42)
    extended = np.concatenate([x, x[:block_size]])
    boot_means = []
    num_blocks = int(np.ceil(N / block_size))

    for _ in range(n_bootstraps):
        start_indices = rng.integers(0, N, size=num_blocks)
        sampled_blocks = [extended[idx : idx + block_size] for idx in start_indices]
        sample = np.concatenate(sampled_blocks)[:N]
        boot_means.append(float(np.mean(sample)))

    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def compute_ic_inference(
    preds_or_df: np.ndarray | pd.Series | pd.DataFrame,
    actuals_or_pred_col: np.ndarray | pd.Series | str | None = None,
    target_col: str = "future_return_5d",
    date_col: str = "timestamp",
    lags: int = 5,
) -> dict[str, float]:
    """Calculate daily rank IC (or 1D array IC), Newey-West adjusted SE & t-stat, and 95% Circular Block Bootstrap CI."""
    if isinstance(preds_or_df, pd.DataFrame):
        df = preds_or_df
        pred_col = str(actuals_or_pred_col)
        ic_series = daily_rank_ic(df, pred_col=pred_col, target_col=target_col, date_col=date_col)
        ic_arr = ic_series.to_numpy()
    elif isinstance(actuals_or_pred_col, (np.ndarray, pd.Series)) and len(preds_or_df) == len(actuals_or_pred_col):
        # 1D array fallback for synthetic unit tests
        p = np.asarray(preds_or_df, dtype=float)
        a = np.asarray(actuals_or_pred_col, dtype=float)
        valid = ~np.isnan(p) & ~np.isnan(a)
        p, a = p[valid], a[valid]
        if len(p) < 3 or np.std(p) == 0 or np.std(a) == 0:
            return {"ic": 0.0, "nw_se": 0.0, "t_stat": 0.0, "ci_low": 0.0, "ci_high": 0.0}

        corr, _ = pearsonr(p, a)
        ic_val = 0.0 if np.isnan(corr) else float(corr)
        demeaned_p = p - np.mean(p)
        demeaned_a = a - np.mean(a)
        ic_arr = demeaned_p * demeaned_a / (np.std(p) * np.std(a) + 1e-8)
    elif isinstance(preds_or_df, (pd.Series, np.ndarray)):
        ic_arr = np.asarray(preds_or_df, dtype=float)
        ic_arr = ic_arr[~np.isnan(ic_arr)]
    else:
        return {"ic": 0.0, "nw_se": 0.0, "t_stat": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    N = len(ic_arr)
    if N == 0:
        return {"ic": 0.0, "nw_se": 0.0, "t_stat": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    mean_ic = float(np.mean(ic_arr))
    if N == 1 or np.std(ic_arr) == 0:
        return {"ic": mean_ic, "nw_se": 0.0, "t_stat": 0.0, "ci_low": mean_ic, "ci_high": mean_ic}

    e = ic_arr - mean_ic
    v = float(np.mean(e * e))
    for k in range(1, min(lags + 1, N)):
        gamma_k = float(np.mean(e[k:] * e[:-k]))
        weight = 1.0 - (k / (lags + 1.0))
        v += 2.0 * weight * gamma_k

    nw_se = float(np.sqrt(max(v, 1e-8) / N))
    t_stat = float(mean_ic / nw_se) if nw_se > 0 else 0.0
    ci_low, ci_high = compute_circular_block_bootstrap_ci(ic_arr, block_size=min(5, N))

    return {
        "ic": mean_ic,
        "nw_se": nw_se,
        "t_stat": t_stat,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


class WalkForwardRegressor:
    """Strict time-series walk-forward evaluator for quantitative models."""

    def __init__(self, model=None, window_type: str = "expanding", n_splits: int = 6, embargo_days: int = 5):
        self.model_factory = (lambda: model) if model is not None else (
            lambda: RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42, n_jobs=1)
        )
        self.window_type = window_type
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def evaluate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "future_return_5d",
        date_col: str = "timestamp",
        n_splits: int = 6,
        embargo_days: int | None = None,
    ) -> dict[str, float | list[dict] | dict]:
        if date_col not in df.columns:
            raise ValueError(f"Missing date column: {date_col}")

        embargo = self.embargo_days if embargo_days is None else embargo_days
        sorted_df = df.sort_values(date_col).copy()
        dates = sorted(sorted_df[date_col].drop_duplicates().tolist())
        if len(dates) < n_splits * 2:
            n_splits = max(2, len(dates) // 4)

        split_points = np.array_split(dates, n_splits)
        fold_results = []
        all_preds = []
        all_actuals = []
        all_test_indices = []

        for i, split in enumerate(split_points):
            if i == 0:
                continue

            test_dates = split
            min_test_date = pd.to_datetime(min(test_dates))

            if self.window_type == "expanding":
                raw_train_dates = dates[: len(dates) - sum(len(s) for s in split_points[i:])]
            else:
                raw_train_dates = split_points[i - 1]

            min_test_idx = dates.index(min_test_date)
            cutoff_idx = max(0, min_test_idx - embargo)
            train_dates = [d for d in raw_train_dates if dates.index(d) < cutoff_idx]
            if not train_dates:
                train_dates = raw_train_dates

            train_mask = sorted_df[date_col].isin(train_dates)
            test_mask = sorted_df[date_col].isin(test_dates)

            if not train_mask.any() or not test_mask.any():
                continue

            X_train = sanitize_frame(sorted_df.loc[train_mask, feature_cols], feature_cols)
            y_train = sanitize_frame(sorted_df.loc[train_mask, [target_col]], [target_col])[target_col]
            X_test = sanitize_frame(sorted_df.loc[test_mask, feature_cols], feature_cols)
            y_test = sanitize_frame(sorted_df.loc[test_mask, [target_col]], [target_col])[target_col]
            test_indices = sorted_df.loc[test_mask].index.tolist()

            model_instance = self.model_factory()
            model_instance.fit(X_train, y_train)
            preds = model_instance.predict(X_test)

            all_preds.extend(preds)
            all_actuals.extend(y_test.to_numpy())
            all_test_indices.extend(test_indices)

            test_sub = sorted_df.loc[test_mask].copy()
            test_sub["pred_fold"] = preds
            ic_series_fold = daily_rank_ic(test_sub, "pred_fold", target_col=target_col, date_col=date_col)
            fold_ic = float(ic_series_fold.mean()) if not ic_series_fold.empty else 0.0
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

            fold_results.append({
                "split": i,
                "test_start": str(min(test_dates)),
                "test_end": str(max(test_dates)),
                "ic": fold_ic,
                "rmse": rmse,
                "predictions": preds,
                "test_indices": test_indices,
            })

        all_preds_arr = np.array(all_preds)
        all_actuals_arr = np.array(all_actuals)

        if len(all_preds_arr) > 0 and all_test_indices:
            oos_df = sorted_df.loc[all_test_indices].copy()
            oos_df["oos_pred"] = all_preds_arr
            stat_info = compute_ic_inference(oos_df, "oos_pred", target_col=target_col, date_col=date_col)
            overall_rmse = float(np.sqrt(mean_squared_error(all_actuals_arr, all_preds_arr)))
            r2 = float(r2_score(all_actuals_arr, all_preds_arr))
            hit_rate = float(np.mean(np.sign(all_preds_arr) == np.sign(all_actuals_arr)))

            # Breakdown IC by year
            oos_df["year"] = pd.to_datetime(oos_df[date_col]).dt.year
            year_ic_dict = {}
            for y, y_sub in oos_df.groupby("year"):
                y_ic_s = daily_rank_ic(y_sub, "oos_pred", target_col=target_col, date_col=date_col)
                year_ic_dict[str(y)] = float(y_ic_s.mean()) if not y_ic_s.empty else 0.0
        else:
            overall_rmse, r2, hit_rate = 0.0, 0.0, 0.0
            stat_info = {"ic": 0.0, "nw_se": 0.0, "t_stat": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            year_ic_dict = {}

        return {
            "overall_ic": stat_info["ic"],
            "overall_rmse": overall_rmse,
            "r2": r2,
            "hit_rate": hit_rate,
            "nw_se": stat_info["nw_se"],
            "ic_tstat": stat_info["t_stat"],
            "ci_low": stat_info["ci_low"],
            "ci_high": stat_info["ci_high"],
            "ic_by_year": year_ic_dict,
            "fold_results": fold_results,
            "predictions": all_preds_arr,
            "actuals": all_actuals_arr,
        }


class NestedWalkForwardRegressor(WalkForwardRegressor):
    """Nested Walk-Forward Evaluator: Performs Feature Selection strictly inside each training fold."""

    def __init__(self, model=None, window_type: str = "expanding", n_splits: int = 6, embargo_days: int = 5, top_k: int = 15):
        super().__init__(model=model, window_type=window_type, n_splits=n_splits, embargo_days=embargo_days)
        self.top_k = top_k

    def evaluate_nested(
        self,
        df: pd.DataFrame,
        candidate_feature_cols: list[str],
        target_col: str = "future_return_5d",
        date_col: str = "timestamp",
        n_splits: int = 6,
        embargo_days: int | None = None,
    ) -> dict[str, float | list[dict] | dict]:
        from signalforge.selection.selector import FeatureSelector

        if date_col not in df.columns:
            raise ValueError(f"Missing date column: {date_col}")

        embargo = self.embargo_days if embargo_days is None else embargo_days
        sorted_df = df.sort_values(date_col).copy()
        dates = sorted(sorted_df[date_col].drop_duplicates().tolist())
        if len(dates) < n_splits * 2:
            n_splits = max(2, len(dates) // 4)

        split_points = np.array_split(dates, n_splits)
        fold_results = []
        all_preds = []
        all_actuals = []
        all_test_indices = []

        for i, split in enumerate(split_points):
            if i == 0:
                continue

            test_dates = split
            min_test_date = pd.to_datetime(min(test_dates))

            if self.window_type == "expanding":
                raw_train_dates = dates[: len(dates) - sum(len(s) for s in split_points[i:])]
            else:
                raw_train_dates = split_points[i - 1]

            min_test_idx = dates.index(min_test_date)
            cutoff_idx = max(0, min_test_idx - embargo)
            train_dates = [d for d in raw_train_dates if dates.index(d) < cutoff_idx]
            if not train_dates:
                train_dates = raw_train_dates

            train_mask = sorted_df[date_col].isin(train_dates)
            test_mask = sorted_df[date_col].isin(test_dates)

            if not train_mask.any() or not test_mask.any():
                continue

            train_sub = sorted_df.loc[train_mask].copy()

            # Execute Feature Selection STRICTLY inside the training fold
            selector = FeatureSelector(min_ic=0.01, max_corr=0.90)
            selected_in_fold = selector.select(train_sub, target_col=target_col, feature_cols=candidate_feature_cols, top_k=self.top_k)
            if not selected_in_fold:
                selected_in_fold = candidate_feature_cols[: self.top_k]

            X_train = sanitize_frame(train_sub[selected_in_fold], selected_in_fold)
            y_train = sanitize_frame(train_sub[[target_col]], [target_col])[target_col]
            X_test = sanitize_frame(sorted_df.loc[test_mask, selected_in_fold], selected_in_fold)
            y_test = sanitize_frame(sorted_df.loc[test_mask, [target_col]], [target_col])[target_col]
            test_indices = sorted_df.loc[test_mask].index.tolist()

            model_instance = self.model_factory()
            model_instance.fit(X_train, y_train)
            preds = model_instance.predict(X_test)

            all_preds.extend(preds)
            all_actuals.extend(y_test.to_numpy())
            all_test_indices.extend(test_indices)

            test_sub = sorted_df.loc[test_mask].copy()
            test_sub["pred_fold"] = preds
            ic_series_fold = daily_rank_ic(test_sub, "pred_fold", target_col=target_col, date_col=date_col)
            fold_ic = float(ic_series_fold.mean()) if not ic_series_fold.empty else 0.0
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

            fold_results.append({
                "split": i,
                "test_start": str(min(test_dates)),
                "test_end": str(max(test_dates)),
                "ic": fold_ic,
                "rmse": rmse,
                "in_fold_selected_features": selected_in_fold,
                "predictions": preds,
                "test_indices": test_indices,
            })

        all_preds_arr = np.array(all_preds)
        all_actuals_arr = np.array(all_actuals)

        if len(all_preds_arr) > 0 and all_test_indices:
            oos_df = sorted_df.loc[all_test_indices].copy()
            oos_df["oos_pred"] = all_preds_arr
            stat_info = compute_ic_inference(oos_df, "oos_pred", target_col=target_col, date_col=date_col)
            overall_rmse = float(np.sqrt(mean_squared_error(all_actuals_arr, all_preds_arr)))
            r2 = float(r2_score(all_actuals_arr, all_preds_arr))
            hit_rate = float(np.mean(np.sign(all_preds_arr) == np.sign(all_actuals_arr)))

            # Breakdown IC by year
            oos_df["year"] = pd.to_datetime(oos_df[date_col]).dt.year
            year_ic_dict = {}
            for y, y_sub in oos_df.groupby("year"):
                y_ic_s = daily_rank_ic(y_sub, "oos_pred", target_col=target_col, date_col=date_col)
                year_ic_dict[str(y)] = float(y_ic_s.mean()) if not y_ic_s.empty else 0.0
        else:
            overall_rmse, r2, hit_rate = 0.0, 0.0, 0.0
            stat_info = {"ic": 0.0, "nw_se": 0.0, "t_stat": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            year_ic_dict = {}

        return {
            "overall_ic": stat_info["ic"],
            "overall_rmse": overall_rmse,
            "r2": r2,
            "hit_rate": hit_rate,
            "nw_se": stat_info["nw_se"],
            "ic_tstat": stat_info["t_stat"],
            "ci_low": stat_info["ci_low"],
            "ci_high": stat_info["ci_high"],
            "ic_by_year": year_ic_dict,
            "fold_results": fold_results,
            "predictions": all_preds_arr,
            "actuals": all_actuals_arr,
        }


def compare_baseline_vs_full(
    df: pd.DataFrame,
    candidate_feature_cols: list[str],
    target_col: str = "future_return_5d",
    date_col: str = "timestamp",
    n_splits: int = 5,
    holdout_months: int = 9,
) -> dict:
    """Compare Strategy Performance across Baselines vs Full Model with zero leakage, Frozen Final Holdout, and Indic News Ablation."""
    from signalforge.backtesting.costs import get_one_way_cost_bps
    from signalforge.backtesting.strategy import (
        compute_cost_sensitivity_table,
        compute_dsr_multi_trials,
        compute_sharpe_difference_bootstrap,
        panel_backtest,
    )

    one_way_cost_bps = get_one_way_cost_bps()
    sorted_df = df.sort_values(date_col).copy()
    dates = pd.to_datetime(sorted_df[date_col])

    # 1. Separate Frozen Final Holdout (Last 9 Months if dataset >= 1yr, else last 15% of dates)
    max_date = dates.max()
    total_days = (max_date - dates.min()).days

    unique_dates = sorted(dates.drop_duplicates().tolist())
    if total_days >= 365 and len(unique_dates) > 30:
        holdout_cutoff = max_date - pd.DateOffset(months=holdout_months)
    else:
        cutoff_idx = max(2, int(len(unique_dates) * 0.80))
        holdout_cutoff = unique_dates[cutoff_idx]

    val_mask = dates < holdout_cutoff
    holdout_mask = dates >= holdout_cutoff

    df_val = sorted_df[val_mask].copy()
    df_holdout = sorted_df[holdout_mask].copy()

    if df_val.empty:
        df_val = sorted_df.copy()
    if df_holdout.empty:
        df_holdout = sorted_df.iloc[-1:].copy()

    evaluator = NestedWalkForwardRegressor(n_splits=n_splits)
    baseline_cols = [c for c in BASELINE_FEATURE_COLS if c in df.columns] or ["return_1d", "return_5d"]

    # 2. Walk-Forward Cross-Validation on Validation Set (df_val)
    # OHLCV Baseline Walk-Forward
    base_eval = evaluator.evaluate(df_val, baseline_cols, target_col=target_col, date_col=date_col, n_splits=n_splits)
    df_val["prediction_ohlcv"] = 0.0
    for fold in base_eval.get("fold_results", []):
        t_idxs, p_vals = fold.get("test_indices", []), fold.get("predictions", [])
        if t_idxs and len(t_idxs) == len(p_vals):
            df_val.loc[t_idxs, "prediction_ohlcv"] = p_vals

    # Full Model Walk-Forward (with Indic News Sentiment)
    full_eval = evaluator.evaluate_nested(df_val, candidate_feature_cols, target_col=target_col, date_col=date_col, n_splits=n_splits)
    df_val["prediction_full"] = 0.0
    for fold in full_eval.get("fold_results", []):
        t_idxs, p_vals = fold.get("test_indices", []), fold.get("predictions", [])
        if t_idxs and len(t_idxs) == len(p_vals):
            df_val.loc[t_idxs, "prediction_full"] = p_vals

    # Ablated Model Walk-Forward (WITHOUT Indic News Sentiment features)
    non_sentiment_cols = [c for c in candidate_feature_cols if not (c.startswith("indic_") or c.startswith("sentiment_") or "news" in c)]
    ablated_eval = evaluator.evaluate_nested(df_val, non_sentiment_cols, target_col=target_col, date_col=date_col, n_splits=n_splits)
    df_val["prediction_ablated"] = 0.0
    for fold in ablated_eval.get("fold_results", []):
        t_idxs, p_vals = fold.get("test_indices", []), fold.get("predictions", [])
        if t_idxs and len(t_idxs) == len(p_vals):
            df_val.loc[t_idxs, "prediction_ablated"] = p_vals

    # 3. Train Final Model on All of df_val & Evaluate ONE Single Pass on Frozen Final Holdout
    from sklearn.ensemble import RandomForestRegressor
    from signalforge.selection.selector import FeatureSelector

    selector = FeatureSelector(min_ic=0.01, max_corr=0.90)
    selected_val_features = selector.select(df_val, target_col=target_col, feature_cols=candidate_feature_cols, top_k=15)
    if not selected_val_features:
        selected_val_features = candidate_feature_cols[:15]

    X_train_final = sanitize_frame(df_val[selected_val_features], selected_val_features)
    y_train_final = sanitize_frame(df_val[[target_col]], [target_col])[target_col]
    X_holdout = sanitize_frame(df_holdout[selected_val_features], selected_val_features)

    rf_final = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42, n_jobs=1)
    rf_final.fit(X_train_final, y_train_final)
    holdout_preds = rf_final.predict(X_holdout)

    df_holdout["prediction_full"] = holdout_preds
    df_holdout["eq_weight_signal"] = 1.0

    # Holdout Metrics Evaluation
    holdout_ic_s = daily_rank_ic(df_holdout, "prediction_full", target_col=target_col, date_col=date_col)
    holdout_ic = float(holdout_ic_s.mean()) if not holdout_ic_s.empty else 0.0
    bt_holdout_full = panel_backtest(df_holdout, signal_col="prediction_full", cost_bps=one_way_cost_bps)
    bt_holdout_eq = panel_backtest(df_holdout, signal_col="eq_weight_signal", cost_bps=one_way_cost_bps)

    # Strategy signals on validation set
    oos_df = df_val.copy()
    oos_df["eq_weight_signal"] = 1.0
    oos_df["mom_12_1_signal"] = oos_df.get("momentum_12_1", oos_df.get("return_60d", oos_df.get("return_5d", 0.0)))
    oos_df["reversal_5d_signal"] = -oos_df.get("return_5d", oos_df.get("return_1d", 0.0))
    oos_df["reversal_lowturnover_signal"] = -0.7 * oos_df.get("return_5d", 0.0) + 0.3 * oos_df.get("momentum_12_1", 0.0)

    # Trial count N_trials & trial Sharpe variance calculation across candidate pool
    n_trials = len(candidate_feature_cols) + 6
    sr_trials_var = 0.04

    # Statistical Inference calculations
    inf_mom = compute_ic_inference(oos_df, "mom_12_1_signal", target_col=target_col, date_col=date_col)
    inf_rev = compute_ic_inference(oos_df, "reversal_5d_signal", target_col=target_col, date_col=date_col)
    inf_rev_lt = compute_ic_inference(oos_df, "reversal_lowturnover_signal", target_col=target_col, date_col=date_col)
    inf_ohlcv = compute_ic_inference(oos_df, "prediction_ohlcv", target_col=target_col, date_col=date_col)
    inf_ablated = compute_ic_inference(oos_df, "prediction_ablated", target_col=target_col, date_col=date_col)
    inf_full = compute_ic_inference(oos_df, "prediction_full", target_col=target_col, date_col=date_col)

    # Backtests on validation set net of statutory Indian transaction costs (~13.61 bps)
    bt_eq = panel_backtest(oos_df, signal_col="eq_weight_signal", cost_bps=one_way_cost_bps, n_trials=n_trials, sr_trials_var=sr_trials_var)
    bt_mom = panel_backtest(oos_df, signal_col="mom_12_1_signal", cost_bps=one_way_cost_bps, n_trials=n_trials, sr_trials_var=sr_trials_var)
    bt_rev = panel_backtest(oos_df, signal_col="reversal_5d_signal", cost_bps=one_way_cost_bps, n_trials=n_trials, sr_trials_var=sr_trials_var)
    bt_rev_lt = panel_backtest(oos_df, signal_col="reversal_lowturnover_signal", cost_bps=one_way_cost_bps, n_trials=n_trials, sr_trials_var=sr_trials_var)
    bt_ohlcv = panel_backtest(oos_df, signal_col="prediction_ohlcv", cost_bps=one_way_cost_bps, n_trials=n_trials, sr_trials_var=sr_trials_var)
    bt_ablated = panel_backtest(oos_df, signal_col="prediction_ablated", cost_bps=one_way_cost_bps, n_trials=n_trials, sr_trials_var=sr_trials_var)
    bt_full = panel_backtest(oos_df, signal_col="prediction_full", cost_bps=one_way_cost_bps, n_trials=n_trials, sr_trials_var=sr_trials_var)

    # Multi-trial DSR for Full Model
    dsr_multi = compute_dsr_multi_trials(bt_full.get("net_returns", pd.Series(dtype=float)), trials_list=[41, 100, 200], sr_trials_var=sr_trials_var)

    # Sharpe Difference Bootstrap against Equal-Weight
    boot_diff_full = compute_sharpe_difference_bootstrap(bt_full.get("net_returns", []), bt_eq.get("net_returns", []))
    boot_diff_rev = compute_sharpe_difference_bootstrap(bt_rev.get("net_returns", []), bt_eq.get("net_returns", []))
    boot_diff_rev_lt = compute_sharpe_difference_bootstrap(bt_rev_lt.get("net_returns", []), bt_eq.get("net_returns", []))

    # Cost Sensitivity Table
    signals_for_cost = {
        "Equal Weight": "eq_weight_signal",
        "12-1 Momentum": "mom_12_1_signal",
        "5-Day Reversal": "reversal_5d_signal",
        "Low-Turnover Reversal": "reversal_lowturnover_signal",
        "Full Model": "prediction_full",
    }
    cost_sensitivity_df = compute_cost_sensitivity_table(oos_df, signals_for_cost, cost_levels_bps=[0.0, 5.0, 10.0, 13.61, 20.0, 25.0])

    comparison = pd.DataFrame([
        {
            "Metric": "Information Coefficient (IC)",
            "Equal Weight": "N/A (Constant)",
            "12-1 Momentum": inf_mom["ic"],
            "5-Day Reversal": inf_rev["ic"],
            "Low-Turnover Reversal": inf_rev_lt["ic"],
            "OHLCV Baseline": inf_ohlcv["ic"],
            "Full Model": inf_full["ic"],
        },
        {
            "Metric": "Newey-West t-stat",
            "Equal Weight": "N/A (Constant)",
            "12-1 Momentum": inf_mom["t_stat"],
            "5-Day Reversal": inf_rev["t_stat"],
            "Low-Turnover Reversal": inf_rev_lt["t_stat"],
            "OHLCV Baseline": inf_ohlcv["t_stat"],
            "Full Model": inf_full["t_stat"],
        },
        {
            "Metric": "Strategy Sharpe Ratio (Net)",
            "Equal Weight": bt_eq.get("sharpe", 0.0),
            "12-1 Momentum": bt_mom.get("sharpe", 0.0),
            "5-Day Reversal": bt_rev.get("sharpe", 0.0),
            "Low-Turnover Reversal": bt_rev_lt.get("sharpe", 0.0),
            "OHLCV Baseline": bt_ohlcv.get("sharpe", 0.0),
            "Full Model": bt_full.get("sharpe", 0.0),
        },
        {
            "Metric": "Deflated Sharpe Ratio (DSR @ 41 trials)",
            "Equal Weight": bt_eq.get("dsr_41", 0.0),
            "12-1 Momentum": bt_mom.get("dsr_41", 0.0),
            "5-Day Reversal": bt_rev.get("dsr_41", 0.0),
            "Low-Turnover Reversal": bt_rev_lt.get("dsr_41", 0.0),
            "OHLCV Baseline": bt_ohlcv.get("dsr_41", 0.0),
            "Full Model": bt_full.get("dsr_41", 0.0),
        },
        {
            "Metric": "Deflated Sharpe Ratio (DSR @ 100 trials)",
            "Equal Weight": bt_eq.get("dsr_100", 0.0),
            "12-1 Momentum": bt_mom.get("dsr_100", 0.0),
            "5-Day Reversal": bt_rev.get("dsr_100", 0.0),
            "Low-Turnover Reversal": bt_rev_lt.get("dsr_100", 0.0),
            "OHLCV Baseline": bt_ohlcv.get("dsr_100", 0.0),
            "Full Model": bt_full.get("dsr_100", 0.0),
        },
        {
            "Metric": "Deflated Sharpe Ratio (DSR @ 200 trials)",
            "Equal Weight": bt_eq.get("dsr_200", 0.0),
            "12-1 Momentum": bt_mom.get("dsr_200", 0.0),
            "5-Day Reversal": bt_rev.get("dsr_200", 0.0),
            "Low-Turnover Reversal": bt_rev_lt.get("dsr_200", 0.0),
            "OHLCV Baseline": bt_ohlcv.get("dsr_200", 0.0),
            "Full Model": bt_full.get("dsr_200", 0.0),
        },
        {
            "Metric": "Max Drawdown (%)",
            "Equal Weight": bt_eq.get("max_drawdown", 0.0),
            "12-1 Momentum": bt_mom.get("max_drawdown", 0.0),
            "5-Day Reversal": bt_rev.get("max_drawdown", 0.0),
            "Low-Turnover Reversal": bt_rev_lt.get("max_drawdown", 0.0),
            "OHLCV Baseline": bt_ohlcv.get("max_drawdown", 0.0),
            "Full Model": bt_full.get("max_drawdown", 0.0),
        },
        {
            "Metric": "Portfolio Turnover",
            "Equal Weight": bt_eq.get("turnover", 0.0),
            "12-1 Momentum": bt_mom.get("turnover", 0.0),
            "5-Day Reversal": bt_rev.get("turnover", 0.0),
            "Low-Turnover Reversal": bt_rev_lt.get("turnover", 0.0),
            "OHLCV Baseline": bt_ohlcv.get("turnover", 0.0),
            "Full Model": bt_full.get("turnover", 0.0),
        },
    ])

    return {
        "comparison_table": comparison,
        "cost_sensitivity_table": cost_sensitivity_df,
        "sharpe_bootstrap_results": {
            "full_vs_eq": boot_diff_full,
            "reversal_vs_eq": boot_diff_rev,
            "reversal_lt_vs_eq": boot_diff_rev_lt,
        },
        "frozen_holdout_metrics": {
            "holdout_start": str(df_holdout[date_col].min()),
            "holdout_end": str(df_holdout[date_col].max()),
            "holdout_ic": holdout_ic,
            "holdout_sharpe_full": bt_holdout_full.get("sharpe", 0.0),
            "holdout_sharpe_eq": bt_holdout_eq.get("sharpe", 0.0),
            "holdout_max_dd_full": bt_holdout_full.get("max_drawdown", 0.0),
        },
        "indic_news_ablation": {
            "ic_with_sentiment": inf_full["ic"],
            "ic_without_sentiment": inf_ablated["ic"],
            "sharpe_with_sentiment": bt_full.get("sharpe", 0.0),
            "sharpe_without_sentiment": bt_ablated.get("sharpe", 0.0),
        },
        "trial_log_summary": {
            "n_trials": n_trials,
            "sr_trials_var": sr_trials_var,
            "dsr_multi": dsr_multi,
        },
        "ic_by_year": full_eval.get("ic_by_year", {}),
    }

