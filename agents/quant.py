from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from signalforge.live.artifacts import load_bundle
from signalforge.live.score import score_universe


def score(df: pd.DataFrame, symbol: str, model_dir: str | Path = "models") -> dict[str, Any]:
    """Evaluates target ticker using SignalForge frozen production model bundle."""
    p_model = Path(model_dir)
    if not p_model.exists() or not (p_model / "model.joblib").exists():
        return {
            "symbol": symbol,
            "prediction": 0.0,
            "weight": 0.0,
            "oos_ic": 0.0042,
            "nw_tstat": 0.38,
            "holdout_ic": -0.0292,
            "high_drift": False,
            "error": "Model bundle not found",
        }

    try:
        bundle = load_bundle(p_model)
        signals, drift_reports, high_drift = score_universe(df, bundle)
        
        sym_sig = signals[signals["symbol"] == symbol] if not signals.empty else pd.DataFrame()
        if not sym_sig.empty:
            pred = float(sym_sig["prediction"].iloc[0])
            weight = float(sym_sig["weight"].iloc[0])
        else:
            pred = 0.0
            weight = 0.0

        return {
            "symbol": symbol,
            "prediction": round(pred, 6),
            "weight": round(weight, 4),
            "oos_ic": 0.0042,
            "nw_tstat": 0.38,
            "holdout_ic": -0.0292,
            "high_drift": bool(high_drift),
            "drift_count": len(drift_reports) if drift_reports else 0,
            "error": None,
        }
    except Exception as err:
        return {
            "symbol": symbol,
            "prediction": 0.0,
            "weight": 0.0,
            "oos_ic": 0.0042,
            "nw_tstat": 0.38,
            "holdout_ic": -0.0292,
            "high_drift": False,
            "error": str(err),
        }
