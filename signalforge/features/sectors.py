from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_MAP = {
    "RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Financials", "INFY": "IT",
    "ICICIBANK": "Financials", "HINDUNILVR": "FMCG", "ITC": "FMCG", "SBIN": "Financials",
    "BHARTIARTL": "Telecom", "GODREJCP": "FMCG", "KOTAKBANK": "Financials", "LT": "Infrastructure",
    "AXISBANK": "Financials", "HCLTECH": "IT", "ASIANPAINT": "Consumer", "MARUTI": "Automobile",
    "SUNPHARMA": "Pharma", "TITAN": "Consumer", "BAJFINANCE": "Financials", "ULTRACEMCO": "Materials",
    "ICICIPRULI": "Financials", "NTPC": "Energy", "ONGC": "Energy", "POWERGRID": "Energy",
    "ADANIENT": "Metals", "ADANIPORTS": "Infrastructure", "COALINDIA": "Energy", "TATASTEEL": "Metals",
    "JSWSTEEL": "Metals", "M&M": "Automobile", "GRASIM": "Materials", "HEROMOTOCO": "Automobile",
    "BAJAJ-AUTO": "Automobile", "EICHERMOT": "Automobile", "BPCL": "Energy", "CIPLA": "Pharma",
    "DRREDDY": "Pharma", "DIVISLAB": "Pharma", "APOLLOHOSP": "Pharma", "TATACONSUM": "FMCG",
    "BRITANNIA": "FMCG", "PIDILITIND": "Materials", "HDFCLIFE": "Financials", "SBILIFE": "Financials",
    "WIPRO": "IT", "TECHM": "IT", "HINDALCO": "Metals", "NESTLEIND": "FMCG",
    "SHRIRAMFIN": "Financials", "BEL": "CapitalGoods"
}


def add_sector_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Add free sector tags to dataframe based on NSE symbol mapping."""
    out = df.copy()
    if "symbol" in out.columns:
        out["sector"] = out["symbol"].map(SECTOR_MAP).fillna("Other")
    else:
        out["sector"] = "Other"
    return out


def compute_sector_neutral_targets(
    df: pd.DataFrame,
    target_col: str = "future_return_5d",
    date_col: str = "timestamp",
) -> pd.DataFrame:
    """Compute sector-demeaned target returns to create sector-neutral prediction targets."""
    out = add_sector_tags(df)
    if date_col in out.columns and target_col in out.columns and "sector" in out.columns:
        sector_means = out.groupby([date_col, "sector"])[target_col].transform("mean")
        out[f"{target_col}_sn"] = out[target_col] - sector_means
    return out


def demean_features_by_sector(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "timestamp",
) -> pd.DataFrame:
    """Cross-sectionally demean features by sector to enforce sector-neutral feature inputs."""
    out = add_sector_tags(df)
    if date_col in out.columns and "sector" in out.columns:
        for col in feature_cols:
            if col in out.columns and np.issubdtype(out[col].dtype, np.number):
                sec_means = out.groupby([date_col, "sector"])[col].transform("mean")
                out[f"{col}_sn"] = out[col] - sec_means
    return out
