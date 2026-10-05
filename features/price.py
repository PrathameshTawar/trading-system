import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rolling_up = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rolling_down = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = rolling_up / rolling_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.fillna(0.0), signal_line.fillna(0.0), hist.fillna(0.0)


def _stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_window, min_periods=1).min()
    highest_high = high.rolling(k_window, min_periods=1).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100 * ((close - lowest_low) / denom)
    stoch_d = stoch_k.rolling(d_window, min_periods=1).mean()
    return stoch_k.fillna(50.0), stoch_d.fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean().fillna(0.0)


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-window returns, momentum, volatility, volume, and scale-free structural signals."""
    out = df.sort_values(["symbol", "timestamp"]).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    if "close" not in out.columns:
        raise ValueError("Price features require a 'close' column.")

    if "open" not in out.columns:
        out["open"] = out["close"]
    if "high" not in out.columns:
        out["high"] = out["close"]
    if "low" not in out.columns:
        out["low"] = out["close"]

    grouped = out.groupby("symbol", group_keys=False)

    # Returns across multiple time horizons
    out["return_1d"] = grouped["close"].pct_change(1).fillna(0.0)
    out["return_5d"] = grouped["close"].pct_change(5).fillna(0.0)
    out["return_10d"] = grouped["close"].pct_change(10).fillna(0.0)
    out["return_20d"] = grouped["close"].pct_change(20).fillna(0.0)
    out["return_60d"] = grouped["close"].pct_change(60).fillna(0.0)

    # Scale-free Moving Averages & Price Distance
    for w in [5, 20, 50, 200]:
        sma = grouped["close"].transform(lambda s, window=w: s.rolling(window, min_periods=1).mean())
        out[f"sma_{w}"] = sma
        out[f"distance_from_sma_{w}"] = (out["close"] / sma.replace(0, np.nan) - 1.0).fillna(0.0)
        out[f"norm_sma_{w}"] = out[f"distance_from_sma_{w}"]

    # Scale-free Volatility (Standard deviation of returns rather than raw price)
    for w in [5, 10, 20, 60]:
        out[f"std_{w}"] = grouped["return_1d"].transform(lambda s, window=w: s.rolling(window, min_periods=2).std().fillna(0.0))
        out[f"norm_std_{w}"] = out[f"std_{w}"]

    # Downside volatility of returns
    def _downside_vol(series: pd.Series, window: int = 20) -> pd.Series:
        rets = series.pct_change()
        neg_rets = rets.clip(upper=0)
        return neg_rets.rolling(window, min_periods=2).std().fillna(0.0)

    out["downside_volatility_20"] = grouped["close"].transform(_downside_vol)

    # Technical momentum signals
    out["rsi_14"] = grouped["close"].transform(lambda s: _rsi(s, 14))

    # Per symbol calculations to guarantee index alignment
    macd_list, signal_list, hist_list = [], [], []
    atr_list, stoch_k_list, stoch_d_list = [], [], []
    obv_list, vpt_list = [], []

    for _, g in grouped:
        m, s, h = _macd(g["close"])
        macd_list.append(m)
        signal_list.append(s)
        hist_list.append(h)

        atr_val = _atr(g["high"], g["low"], g["close"], 14)
        atr_list.append(atr_val)

        sk, sd = _stochastic_oscillator(g["high"], g["low"], g["close"])
        stoch_k_list.append(sk)
        stoch_d_list.append(sd)

        if "volume" in g.columns:
            sign = np.sign(g["close"].diff()).fillna(0.0)
            obv_list.append((sign * g["volume"]).cumsum())
            pct = g["close"].pct_change().fillna(0.0)
            vpt_list.append((pct * g["volume"]).cumsum())

    out["macd"] = pd.concat(macd_list)
    out["macd_signal"] = pd.concat(signal_list)
    out["macd_hist"] = pd.concat(hist_list)

    raw_atr = pd.concat(atr_list)
    out["atr_14"] = raw_atr
    out["norm_atr_14"] = (raw_atr / out["close"].replace(0, np.nan)).fillna(0.0)
    out["stoch_k"] = pd.concat(stoch_k_list)
    out["stoch_d"] = pd.concat(stoch_d_list)

    # Rate of Change (ROC)
    out["roc_5"] = grouped["close"].transform(lambda s: (s.pct_change(5) * 100).fillna(0.0))
    out["roc_10"] = grouped["close"].transform(lambda s: (s.pct_change(10) * 100).fillna(0.0))
    out["roc_20"] = grouped["close"].transform(lambda s: (s.pct_change(20) * 100).fillna(0.0))

    out["momentum_5"] = out["return_5d"]
    out["momentum_10"] = out["return_10d"]
    out["momentum_20"] = out["return_20d"]
    out["momentum_12_1"] = grouped["close"].transform(lambda s: (s.shift(21) / s.shift(252).replace(0, np.nan) - 1.0).fillna(0.0))


    # Price extremes (52-week equivalent: 252 periods or max available)
    high_52w = grouped["high"].transform(lambda s: s.rolling(252, min_periods=1).max())
    low_52w = grouped["low"].transform(lambda s: s.rolling(252, min_periods=1).min())
    out["distance_from_52w_high"] = (out["close"] / high_52w.replace(0, np.nan) - 1.0).fillna(0.0)
    out["distance_from_52w_low"] = (out["close"] / low_52w.replace(0, np.nan) - 1.0).fillna(0.0)

    # Leak-free rolling volume features
    if "volume" in out.columns:
        out["volume_change"] = grouped["volume"].pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vol_mean = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        vol_std = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=5).std().replace(0, np.nan))
        out["volume_zscore"] = ((out["volume"] - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if obv_list:
            out["obv"] = pd.concat(obv_list)
        if vpt_list:
            raw_vpt = pd.concat(vpt_list)
            out["volume_price_trend"] = raw_vpt
            vpt_shifted = grouped["volume_price_trend"].transform(lambda s: s.shift(20))
            out["norm_vpt_20d"] = ((raw_vpt - vpt_shifted) / (vpt_shifted.abs() + 1e-6)).fillna(0.0)

    # Target variable (Forward 5-day return aligned with Next-Open execution: open_t+6 / open_t+1 - 1.0)
    open_fwd = grouped["open"].shift(-6) / grouped["open"].shift(-1) - 1.0
    out["future_return_5d"] = open_fwd
    return out

