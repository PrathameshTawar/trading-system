from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def scores_to_cash_weights(
    signals: pd.DataFrame,
    *,
    long_only: bool = True,
    max_names: int = 10,
    max_weight: float = 0.05,
    gross_exposure: float = 1.0,
    min_score: float = -999.0,
    current_positions: set[str] | dict[str, Any] | None = None,
    buffer_pct: float = 0.30,
) -> pd.DataFrame:
    """Map model predictions to equal-weighted cash-equity target weights for top quintile names.

    Includes position cap (default 5%) and hysteresis holding buffer (holds existing names
    unless they drop below top 30% rank).
    """
    out = signals.copy()
    if out.empty or "prediction" not in out.columns:
        out["weight"] = pd.Series(0.0, index=out.index)
        return out

    # Convert current_positions to a set of symbols
    held_syms: set[str] = set()
    if isinstance(current_positions, set):
        held_syms = current_positions
    elif isinstance(current_positions, dict):
        held_syms = set(current_positions.keys())

    scores = out["prediction"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["pred_rank"] = scores.rank(ascending=False, method="min")

    n_universe = len(out)
    top_k = min(max_names, max(1, int(n_universe * 0.20))) if n_universe >= 5 else min(max_names, n_universe)
    buffer_rank_limit = max(top_k, int(n_universe * buffer_pct)) if n_universe >= 5 else top_k

    # Select candidates:
    # 1. Top-K fresh entries
    top_candidates = out[out["pred_rank"] <= top_k]["symbol"].tolist()

    # 2. Retain held names if their rank is still within buffer limit (e.g. top 30%)
    retained_held = [
        sym for sym in held_syms
        if sym in out["symbol"].values and out.loc[out["symbol"] == sym, "pred_rank"].values[0] <= buffer_rank_limit
    ]

    selected_syms = sorted(list(set(top_candidates + retained_held)))[:top_k]
    if not selected_syms:
        selected_syms = top_candidates[:top_k]

    weights = pd.Series(0.0, index=out.index)
    if selected_syms:
        per_name_weight = min(max_weight, gross_exposure / max(len(selected_syms), 1))
        mask = out["symbol"].isin(selected_syms)
        weights.loc[mask] = per_name_weight

    out["weight"] = weights.clip(lower=0.0, upper=max_weight)
    if "pred_rank" in out.columns:
        out = out.drop(columns=["pred_rank"])
    return out
