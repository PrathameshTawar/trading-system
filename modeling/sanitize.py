from __future__ import annotations

import numpy as np
import pandas as pd

# sklearn trees store features as float32; values beyond this abort fit.
_FLOAT32_CAP = 1.0e30


def sanitize_frame(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Replace inf/NaN and clip extreme values so tree models can fit real market data."""
    out = df.copy()
    if columns is None:
        columns = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for col in columns:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        values = np.nan_to_num(values, nan=0.0, posinf=_FLOAT32_CAP, neginf=-_FLOAT32_CAP)
        values = np.clip(values, -_FLOAT32_CAP, _FLOAT32_CAP)
        out[col] = values
    return out

