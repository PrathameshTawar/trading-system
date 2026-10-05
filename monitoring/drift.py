from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between reference and actual feature distributions."""
    exp = expected.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    act = actual.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(exp) == 0 or len(act) == 0:
        return 0.0

    expected_bins = np.linspace(exp.min(), exp.max(), bins + 1)
    expected_bins[0] = -np.inf
    expected_bins[-1] = np.inf

    exp_dist = np.histogram(exp, bins=expected_bins)[0] / max(len(exp), 1)
    act_dist = np.histogram(act, bins=expected_bins)[0] / max(len(act), 1)

    epsilon = 1e-6
    exp_dist = np.clip(exp_dist, epsilon, None)
    act_dist = np.clip(act_dist, epsilon, None)

    return float(np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist)))


def compute_ks_test(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Compute Kolmogorov-Smirnov statistic and p-value for distribution shift."""
    ref_clean = reference.replace([np.inf, -np.inf], np.nan).dropna()
    cur_clean = current.replace([np.inf, -np.inf], np.nan).dropna()

    if len(ref_clean) < 5 or len(cur_clean) < 5:
        return 0.0, 1.0

    stat, p_val = ks_2samp(ref_clean, cur_clean)
    return float(stat), float(p_val)


def compute_drift_report(reference: pd.Series, current: pd.Series, feature_name: str = "feature") -> dict[str, str | float]:
    """Compute full drift report including PSI, KS-test, mean/std shift, and alert status."""
    ref_mean = float(reference.mean()) if len(reference) > 0 else 0.0
    cur_mean = float(current.mean()) if len(current) > 0 else 0.0

    ref_std = float(reference.std(ddof=1)) if len(reference) > 1 else 1.0
    cur_std = float(current.std(ddof=1)) if len(current) > 1 else 1.0

    psi_val = psi(reference, current)
    ks_stat, ks_pval = compute_ks_test(reference, current)

    missing_ref = float(reference.isna().mean())
    missing_cur = float(current.isna().mean())

    if psi_val >= 0.25 or ks_pval < 0.01:
        status = "HIGH"
        recommended_action = "Investigate data feed / Retrain model immediately"
    elif psi_val >= 0.10 or ks_pval < 0.05:
        status = "MEDIUM"
        recommended_action = "Monitor closely / Schedule retrain"
    else:
        status = "LOW"
        recommended_action = "No action required"

    return {
        "feature": feature_name,
        "mean_change": float(cur_mean - ref_mean),
        "std_change": float(cur_std - ref_std),
        "missing_ratio_change": float(missing_cur - missing_ref),
        "psi": psi_val,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pval,
        "status": status,
        "recommended_action": recommended_action,
    }


class DriftMonitor:
    """Monitor feature matrices for distribution drift over time."""

    def __init__(self, reference_df: pd.DataFrame):
        self.reference_df = reference_df

    def check_drift(self, current_df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
        if feature_cols is None:
            feature_cols = [c for c in current_df.columns if c in self.reference_df.columns and c not in {"timestamp", "symbol"}]

        reports = []
        for col in feature_cols:
            if col in self.reference_df.columns and col in current_df.columns:
                rep = compute_drift_report(self.reference_df[col], current_df[col], feature_name=col)
                reports.append(rep)

        return pd.DataFrame(reports)


def drift_status(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    min_rows: int = 100,
    psi_hi: float = 0.25,
    min_breaches: int = 3,
) -> str:
    """Robust multi-feature drift gate preventing sample-size false alarms.

    Requires at least min_rows (e.g. 100+ rows across 20-60 sessions) to judge drift,
    and requires at least min_breaches (e.g. 3 features) to breach psi_hi (0.25)
    simultaneously before returning 'HIGH'.
    """
    if current_df is None or len(current_df) < min_rows:
        return "SKIP"  # Too little data to judge; do not flatten the book

    breaches = 0
    for feat in feature_cols:
        if feat in reference_df.columns and feat in current_df.columns:
            psi_val = psi(reference_df[feat], current_df[feat])
            if psi_val > psi_hi:
                breaches += 1

    return "HIGH" if breaches >= min_breaches else "OK"

