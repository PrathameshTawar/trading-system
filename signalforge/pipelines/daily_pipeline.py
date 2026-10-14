from __future__ import annotations

import pandas as pd

from signalforge.backtesting.strategy import run_strategy_backtest
from signalforge.features.factory import FeatureFactory
from signalforge.modeling.walk_forward import WalkForwardRegressor, compare_baseline_vs_full
from signalforge.monitoring.drift import compute_drift_report
from signalforge.selection.selector import FeatureSelector
from signalforge.validation.leakage import LeakageValidator
from signalforge.validation.quality import QualityFilter


class DailyPipeline:
    """End-to-end production daily quantitative pipeline."""

    def __init__(
        self,
        target_col: str = "future_return_5d",
        timestamp_col: str = "timestamp",
        config_path: str | None = None,
        selected_limit: int = 15,
    ):
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.selected_limit = selected_limit
        self.factory = FeatureFactory(config_path=config_path)
        self.selector = FeatureSelector(min_ic=0.01, max_corr=0.90)
        self.quality_filter = QualityFilter()
        self.validator = LeakageValidator(timestamp_col=timestamp_col, target_col=target_col)

    def run(self, df: pd.DataFrame, feature_lag_map: dict[str, int] | None = None, reference_df: pd.DataFrame | None = None) -> dict:
        # 1. Feature Generation
        features_df = self.factory.generate(df)
        all_candidate_cols = self.factory.get_feature_names(features_df)

        # 2. Quality & Leakage Validation
        clean_feature_cols, rejections = self.quality_filter.filter_features(features_df, all_candidate_cols)
        leakage_issues = self.validator.validate(features_df, feature_lag_map or {})

        # 3. Multi-Stage IC & Intelligence Selection
        eval_table = self.selector.evaluate_features(features_df, target_col=self.target_col, feature_cols=clean_feature_cols)
        selected_features = self.selector.select(features_df, target_col=self.target_col, feature_cols=clean_feature_cols, top_k=self.selected_limit)

        if not selected_features:
            selected_features = clean_feature_cols[: self.selected_limit]

        # 4. Walk-Forward Baseline vs Full Model Evaluation (with Frozen Holdout & Indic News Ablation)
        baseline_vs_full = compare_baseline_vs_full(features_df, clean_feature_cols, target_col=self.target_col, date_col=self.timestamp_col, n_splits=5)

        # Extract metrics
        comparison_records = (
            baseline_vs_full["comparison_table"].to_dict(orient="records")
            if isinstance(baseline_vs_full, dict) and "comparison_table" in baseline_vs_full
            else []
        )

        full_ic = 0.0
        full_tstat = 0.0
        for r in comparison_records:
            if r.get("Metric") == "Information Coefficient (IC)":
                full_ic = float(r.get("Full Model", 0.0)) if isinstance(r.get("Full Model"), (int, float)) else 0.0
            elif r.get("Metric") == "Newey-West t-stat":
                full_tstat = float(r.get("Full Model", 0.0)) if isinstance(r.get("Full Model"), (int, float)) else 0.0

        backtest_results = run_strategy_backtest(features_df, signal_col="prediction", return_col=self.target_col)

        # 5. Feature Drift Monitoring
        drift_reports = []
        if reference_df is not None:
            for feat in selected_features[:5]:
                if feat in reference_df.columns and feat in features_df.columns:
                    rep = compute_drift_report(reference_df[feat], features_df[feat], feature_name=feat)
                    drift_reports.append(rep)

        return {
            "total_candidate_features": len(all_candidate_cols),
            "quality_filtered_features": len(clean_feature_cols),
            "rejection_reasons": rejections,
            "selected_features": selected_features,
            "walkforward": baseline_vs_full.get("fold_results", []),
            "leakage_issues": leakage_issues,
            "evaluation_ranking": eval_table.head(10).to_dict(orient="records") if not eval_table.empty else [],
            "walk_forward_metrics": {
                "ic": full_ic,
                "ic_tstat": full_tstat,
                "ic_ci_low": (
                    baseline_vs_full.get("full_model_ic_inference", {}).get("ci_low")
                    if isinstance(baseline_vs_full, dict) and "full_model_ic_inference" in baseline_vs_full
                    else (full_ic - 1.96 * (full_ic / full_tstat) if full_tstat != 0 else -0.05)
                ),
                "ic_ci_high": (
                    baseline_vs_full.get("full_model_ic_inference", {}).get("ci_high")
                    if isinstance(baseline_vs_full, dict) and "full_model_ic_inference" in baseline_vs_full
                    else (full_ic + 1.96 * (full_ic / full_tstat) if full_tstat != 0 else 0.05)
                ),
            },
            "baseline_vs_full_comparison": comparison_records,
            "cost_sensitivity_table": (
                baseline_vs_full["cost_sensitivity_table"].to_dict(orient="records")
                if isinstance(baseline_vs_full, dict) and "cost_sensitivity_table" in baseline_vs_full
                else []
            ),
            "sharpe_bootstrap_results": baseline_vs_full.get("sharpe_bootstrap_results", {}) if isinstance(baseline_vs_full, dict) else {},
            "frozen_holdout_metrics": baseline_vs_full.get("frozen_holdout_metrics", {}) if isinstance(baseline_vs_full, dict) else {},
            "indic_news_ablation": baseline_vs_full.get("indic_news_ablation", {}) if isinstance(baseline_vs_full, dict) else {},
            "trial_log_summary": baseline_vs_full.get("trial_log_summary", {}) if isinstance(baseline_vs_full, dict) else {},
            "ic_by_year": baseline_vs_full.get("ic_by_year", {}) if isinstance(baseline_vs_full, dict) else {},
            "backtest_results": backtest_results,
            "drift_reports": drift_reports,
        }

