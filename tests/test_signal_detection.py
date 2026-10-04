import numpy as np
import pandas as pd

from signalforge.data import generate_synthetic_market_data
from signalforge.features.factory import FeatureFactory
from signalforge.modeling.walk_forward import NestedWalkForwardRegressor, compute_ic_inference
from signalforge.backtesting.strategy import run_strategy_backtest, compute_deflated_sharpe_ratio


def test_planted_signal_recovery():
    """Planted-Signal Recovery Test:

    Inject a known synthetic alpha signal with true IC ≈ +0.10 into market data.
    Verify that the pipeline selects the planted feature in-fold and recovers a positive out-of-sample IC.
    """
    df = generate_synthetic_market_data(days=250, symbols=("SYM1", "SYM2", "SYM3"))
    factory = FeatureFactory()
    features_df = factory.generate(df)

    # Generate a clean planted feature and add strong positive alignment to future_return_5d
    rng = np.random.default_rng(42)
    planted_signal = rng.normal(0, 1.0, size=len(features_df))
    features_df["planted_alpha_signal"] = planted_signal
    valid_mask = features_df["future_return_5d"].notna()
    features_df.loc[valid_mask, "future_return_5d"] += 0.08 * planted_signal[valid_mask]

    candidate_cols = factory.get_feature_names(features_df) + ["planted_alpha_signal"]

    evaluator = NestedWalkForwardRegressor(n_splits=4, top_k=5)
    results = evaluator.evaluate_nested(features_df, candidate_cols, target_col="future_return_5d")

    # 1. Confirm planted signal is selected in training folds
    in_fold_features = [feat for fold in results.get("fold_results", []) for feat in fold.get("in_fold_selected_features", [])]
    assert "planted_alpha_signal" in in_fold_features, f"Pipeline failed to select planted signal: {in_fold_features}"

    # 2. Confirm out-of-sample IC is positive
    overall_ic = results.get("overall_ic", 0.0)
    assert overall_ic > 0.01, f"Planted signal recovery failed: out-of-sample IC = {overall_ic:.4f}"



def test_null_control_shuffled_labels():
    """Null-Control Test:

    Shuffle the target label future_return_5d randomly across rows.
    Verify that out-of-sample IC is near 0.0 and Newey-West t-stat fails to reject null (|t| < 2.0).
    """
    df = generate_synthetic_market_data(days=200, symbols=("SYM1", "SYM2"))
    factory = FeatureFactory()
    features_df = factory.generate(df)

    # Shuffle target labels to destroy any true relationship
    rng = np.random.default_rng(42)
    features_df["future_return_5d"] = rng.permutation(features_df["future_return_5d"].fillna(0.0).values)

    candidate_cols = factory.get_feature_names(features_df)
    evaluator = NestedWalkForwardRegressor(n_splits=4, top_k=5)
    results = evaluator.evaluate_nested(features_df, candidate_cols, target_col="future_return_5d")

    overall_ic = abs(results.get("overall_ic", 0.0))
    t_stat = abs(results.get("ic_tstat", 0.0))

    assert overall_ic < 0.05, f"Null control failed: spurious IC detected = {overall_ic:.4f}"
    assert t_stat < 2.0, f"Null control failed: spurious t-stat detected = {t_stat:.2f}"


def test_newey_west_standard_error_and_bootstrap_ci():
    """Statistical Unit Test:

    Verify Newey-West standard error against statsmodels OLS HAC estimator and circular block bootstrap CI bounds.
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(42)
    N = 250
    actuals = rng.normal(0, 0.02, size=N)
    preds = actuals * 0.3 + rng.normal(0, 0.02, size=N)

    inf = compute_ic_inference(preds, actuals)
    assert "ic" in inf and "nw_se" in inf and "t_stat" in inf and "ci_low" in inf and "ci_high" in inf
    assert inf["ic"] > 0, "IC should be positive for correlated signals"
    assert inf["ci_low"] <= inf["ic"] <= inf["ci_high"], f"Bootstrap CI [{inf['ci_low']:.4f}, {inf['ci_high']:.4f}] must contain mean IC {inf['ic']:.4f}"

    # Reference check against statsmodels OLS HAC (lags=5)
    X = sm.add_constant(preds)
    model = sm.OLS(actuals, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    sm_se = float(model.bse[1])

    assert inf["nw_se"] > 0, "Newey-West SE must be strictly positive"
    assert abs(inf["nw_se"] - sm_se) < 0.05, f"NW SE {inf['nw_se']:.5f} deviates significantly from statsmodels HAC SE {sm_se:.5f}"



def test_deflated_sharpe_ratio_computation():
    """Statistical Unit Test: Verify López de Prado Deflated Sharpe Ratio calculation."""
    rets = pd.Series([0.01, -0.005, 0.012, 0.008, -0.002, 0.015, -0.001, 0.009] * 10)
    dsr = compute_deflated_sharpe_ratio(rets, n_trials=5)
    assert 0.0 <= dsr <= 1.0, f"Deflated Sharpe ratio probability must be in [0, 1]: got {dsr}"
