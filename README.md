# SignalForge: Quantitative Feature Engineering & Portfolio Benchmark Pipeline

SignalForge is an empirical quantitative research framework designed for cash equity markets (NSE / BSE / US Equities). It provides automated scale-free feature engineering, multi-stage IC feature selection, nested walk-forward cross-validation, matrix panel backtesting, and a paper-trading demonstration harness.

---

## 🏛️ Key Methodological Guardrails

1. **Scale-Free Stationary Candidate Pool**:
   - All raw price levels (`sma_5..200`, `ema_*`, `std_5..60`, `atr_14`, `obv`, `volume_price_trend`, `nifty_close`, `adj_close`) are strictly pruned prior to feature selection.
   - Only stationary scale-free indicators (`norm_sma_*`, `norm_std_*`, `norm_atr_*`, `norm_vpt_*`, `distance_from_52w_high/low`, `relative_return_vs_benchmark`, `market_regime`) enter the candidate pool, preventing tree models from memorizing symbol identity.

2. **In-Fold Feature Selection**:
   - `NestedWalkForwardRegressor` executes feature selection strictly inside each training fold, preventing feature selection leakage from test split distributions.

3. **Trading-Day Embargo Purging**:
   - Enforces position-based index purging (`cutoff_idx = min_test_idx - embargo_days`) on sorted trading days, removing overlapping label horizons between train and test splits.

4. **Target Alignment (Next-Open Execution)**:
   - Target variable is strictly defined as forward 5-day open-to-open return:
     $$\text{Target}_t = \frac{\text{Open}_{t+6}}{\text{Open}_{t+1}} - 1.0$$
   - Clean next-open execution without mixing fallback close-to-close targets.

5. **Matrix Panel Backtesting**:
   - Vectorized matrix operations on pivoted $(Timestamp \times Symbol)$ data. Eliminates cross-symbol return compounding artifacts found in stacked long-format DataFrames.

---

## 🔬 Statistical Inference & Proof Tests

- **Newey-West $t$-Statistics**: Autocorrelation-adjusted standard errors with lag 5 (matching overlapping 5-day return horizon).
- **Circular Block Bootstrap CIs**: 95% confidence intervals constructed via Circular Block Bootstrap ($L = 5$ days) to preserve target autocorrelation.
- **Planted-Signal Recovery Test (`test_planted_signal_recovery`)**:
  - Injects a synthetic signal ($IC \approx +0.10$) into features and verifies that the pipeline successfully selects the feature in-fold and recovers a statistically significant out-of-sample IC ($IC > 0.01$).
- **Null-Control Test (`test_null_control_shuffled_labels`)**:
  - Shuffles target labels randomly across date/symbol pairs and verifies that out-of-sample IC evaluates near 0 ($|IC| < 0.05$) and fails to reject the null hypothesis ($|t| < 2.0$).
- **Empirical Trial Log & López de Prado Deflated Sharpe Ratio (DSR)**:
  - Adjusts backtest Sharpe ratios for selection bias, number of evaluated candidate feature trials, and non-normal skewness/kurtosis.

---

## 📊 Benchmark Strategy Comparison (50 Liquid Large Caps, 5 Years)

| Strategy Baseline | Out-of-Sample IC | Newey-West $t$-stat | Strategy Sharpe (Net ~13.6bps) | DSR (41 / 100 / 200 trials) | Max Drawdown (%) | Portfolio Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal-Weight Buy & Hold** | N/A (Constant) | N/A (Constant) | `+1.15` | `0.92` | `-16.76%` | `0.00` |
| **12-1 Month Momentum** | `-0.0013` | `-0.1031` | `+1.01` | `0.88` | `-21.39%` | `0.14` |
| **5-Day Short Reversal** | `0.0390` | `3.9381` | `+0.57` | `0.60` | `-18.18%` | `1.22` |
| **Low-Turnover Reversal** | `0.0190` | `1.7679` | `+0.73` | `0.72` | `-22.04%` | `0.57` |
| **OHLCV Linear Baseline** | `0.0000` | `0.0057` | `+0.61` | `0.62` | `-20.14%` | `1.23` |
| **Full SignalForge Model** | **`0.0042`** | **`0.3834`** | **`+0.95`** | **`0.82 / 0.79 / 0.76`** | **`-20.65%`** | **`0.47`** |

### 🧊 Frozen Final Holdout Evaluation (Last 9 Months)

| Holdout Period | Frozen Holdout IC | Full Model Net Sharpe | Equal Weight Net Sharpe | Max Drawdown |
| :---: | :---: | :---: | :---: | :---: |
| `2026-01-01 00:00:00 to 2026-10-01 00:00:00` | `-0.0292` | `-1.37` | `-0.55` | `-16.01%` |

### 📰 Indic News Sentiment Ablation Study

| Model Variant | Out-of-Sample Rank IC | Net Strategy Sharpe Ratio |
| :--- | :---: | :---: |
| **Full Model WITH Indic News Sentiment** | `+0.0042` | `+0.95` |
| **Ablated Model WITHOUT Sentiment** | `+0.0092` | `+1.08` |

### 💸 Transaction Cost Sensitivity (Sharpe at 0 to 25 bps)

| Transaction Cost Level | Equal Weight | 12-1 Momentum | 5-Day Reversal | Low-Turnover Reversal | Full Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `0.00 bps` | `+1.15` | `+1.09` | `+1.15` | `+0.99` | `+1.20` |
| `5.00 bps` | `+1.15` | `+1.06` | `+0.94` | `+0.89` | `+1.11` |
| `10.00 bps` | `+1.15` | `+1.03` | `+0.72` | `+0.80` | `+1.02` |
| `13.61 bps` | `+1.15` | `+1.01` | `+0.57` | `+0.73` | `+0.95` |
| `20.00 bps` | `+1.15` | `+0.97` | `+0.30` | `+0.61` | `+0.84` |
| `25.00 bps` | `+1.15` | `+0.94` | `+0.08` | `+0.51` | `+0.75` |

### 🎲 Sharpe Difference Circular Block Bootstrap vs Equal-Weight

| Comparison vs Equal-Weight | Sharpe Difference ($\Delta SR$) | 95% Bootstrap CI | $p$-value |
| :--- | :---: | :---: | :---: |
| **Full Model vs Equal Weight** | `-0.03` | `[-1.73, +1.48]` | `0.9600` |
| **5-Day Reversal vs Equal Weight** | `-0.58` | `[-0.96, -0.25]` | `0.0000` |
| **Low-Turnover Reversal vs Equal Weight** | `-0.42` | `[-0.71, -0.15]` | `0.0040` |

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
pip install -e .[dev]
```

### 2. Run Test Suite (24 Unit & Verification Tests)
```bash
python -m pytest -v
```

### 3. Fetch 5-Year NSE EOD & Benchmark Data (50 Constituent Universe)
```bash
python main.py fetch-eod --symbols "RELIANCE,TCS,HDFCBANK,INFY,ICICIBANK,HINDUNILVR,ITC,SBIN,BHARTIARTL,GODREJCP,KOTAKBANK,LT,AXISBANK,HCLTECH,ASIANPAINT,MARUTI,SUNPHARMA,TITAN,BAJFINANCE,ULTRACEMCO,ICICIPRULI,NTPC,ONGC,POWERGRID,ADANIENT,ADANIPORTS,COALINDIA,TATASTEEL,JSWSTEEL,M&M,GRASIM,HEROMOTOCO,BAJAJ-AUTO,EICHERMOT,BPCL,CIPLA,DRREDDY,DIVISLAB,APOLLOHOSP,TATACONSUM,BRITANNIA,PIDILITIND,HDFCLIFE,SBILIFE,WIPRO,TECHM,HINDALCO,NESTLEIND,SHRIRAMFIN,BEL" --period 5y --output data/eod.csv
```

### 4. Execute Quantitative Pipeline & Reproduce Benchmarks
```bash
python reproduce.py
```

### 5. Train Production Model Bundle (Demonstration Integration Harness)
```bash
python main.py train --input data/eod.csv --model-dir models/frozen_v1
```

---

## 📁 Repository Structure

```
signalforge/
├── backtesting/      # Vectorized matrix panel backtester & Deflated Sharpe Ratio
├── features/         # Scale-free price, volume, cross-asset & Indic NLP features
├── ingestion/        # Multi-CSV merger, NSE EOD loader & real ^NSEI benchmark
├── live/             # Production bundle serialization & paper trading demonstration harness
├── modeling/         # Nested walk-forward cross-validation & baseline models
├── monitoring/       # Kolmogorov-Smirnov & PSI feature drift monitoring
├── selection/        # Multi-stage IC, mutual info & redundancy selector
└── validation/       # Quality filter & temporal leakage validator
tests/                # Planted-signal, null-control, truncation & unit test suite
```

---

## 📜 License
Distributed under the MIT License.
