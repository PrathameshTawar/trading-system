import numpy as np
import pandas as pd
from signalforge.monitoring.drift import DriftMonitor, compute_drift_report, psi


def test_psi_calculation():
    ref = pd.Series(np.random.normal(0, 1, 1000))
    cur = pd.Series(np.random.normal(0, 1, 1000))
    val = psi(ref, cur)
    assert val >= 0.0
    assert val < 0.1  # Identical distribution should have low PSI


def test_drift_report_triggers_high_status_on_shift():
    ref = pd.Series(np.random.normal(0, 1, 1000))
    shifted = pd.Series(np.random.normal(5, 1, 1000))  # Significant mean shift
    rep = compute_drift_report(ref, shifted, feature_name="return_1d")
    assert rep["status"] == "HIGH"
    assert rep["psi"] > 0.25


def test_drift_monitor():
    ref_df = pd.DataFrame({"feat_a": np.random.normal(0, 1, 100), "feat_b": np.random.normal(1, 2, 100)})
    cur_df = pd.DataFrame({"feat_a": np.random.normal(0, 1, 100), "feat_b": np.random.normal(10, 2, 100)})
    monitor = DriftMonitor(ref_df)
    res = monitor.check_drift(cur_df)
    assert len(res) == 2
    assert "psi" in res.columns
