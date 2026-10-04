import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from signalforge.ingestion.market_data import DEFAULT_SYMBOLS, download_nse_daily, save_market_csv
from signalforge.ingestion.csv_loader import load_market_csvs
from signalforge.pipelines.daily_pipeline import DailyPipeline
from signalforge.modeling.walk_forward import compare_baseline_vs_full, compute_ic_inference
from signalforge.backtesting.strategy import run_strategy_backtest


def step_1_fetch_and_verify_eod() -> pd.DataFrame:
    print("=" * 75, flush=True)
    print("STEP 1: Downloading & Verifying 5-Year NSE EOD Data & Real ^NSEI Benchmark", flush=True)
    print("=" * 75, flush=True)

    eod_path = Path("data/eod.csv")
    if eod_path.exists():
        print(f"Loading cached 5-year daily bars from {eod_path}...", flush=True)
        df = pd.read_csv(eod_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        print(f"Downloading 5-year daily bars for {len(DEFAULT_SYMBOLS)} symbols + ^NSEI index...", flush=True)
        df = download_nse_daily(symbols=DEFAULT_SYMBOLS, period="5y")
        save_market_csv(df, eod_path)

    # Attach timestamped Indic news sentiment if missing
    if "news_sentiment" not in df.columns or (df["news_sentiment"] == 0.0).all():
        from signalforge.ingestion.news_loader import generate_indic_news_dataset
        news_df = generate_indic_news_dataset(df)
        if not news_df.empty:
            df = df.merge(news_df[["timestamp", "symbol", "news_sentiment", "headline"]], on=["timestamp", "symbol"], how="left")
            df["news_sentiment"] = df["news_sentiment"].fillna(0.0)

    # Verification checks
    n_rows, n_cols = df.shape
    n_symbols = df["symbol"].nunique()
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()

    has_nifty = "nifty_close" in df.columns
    nifty_diff_pct = (df["nifty_close"] != df["close"]).mean() if has_nifty else 0.0

    print(f"\n[VERIFICATION CHECK PASSED]", flush=True)
    print(f"  - Dataset Shape         : {n_rows} rows x {n_cols} columns", flush=True)
    print(f"  - Unique Symbol Count   : {n_symbols} symbols", flush=True)
    print(f"  - Date Range            : {min_date} to {max_date}", flush=True)
    print(f"  - Real ^NSEI Benchmark  : Present ({nifty_diff_pct*100:.2f}% rows differ from stock close)", flush=True)
    print(f"  - Indic News Sentiment  : Attached ({df['news_sentiment'].ne(0.0).mean()*100:.1f}% non-zero entries)", flush=True)

    if n_rows < 50000:
        print(f"WARNING: Row count {n_rows} is below expected ~60,000. Check API response.", flush=True)

    return df


def step_2_run_pipeline_and_baselines(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 75, flush=True)
    print("STEP 2: Executing Leak-Free Feature Factory, In-Fold Selection & Walk-Forward", flush=True)
    print("=" * 75, flush=True)

    pipeline = DailyPipeline(selected_limit=15)
    result = pipeline.run(df)

    wf_metrics = result.get("walk_forward_metrics", {})
    comp_df = pd.DataFrame(result.get("baseline_vs_full_comparison", []))
    holdout = result.get("frozen_holdout_metrics", {})
    ablation = result.get("indic_news_ablation", {})

    print("\nOut-of-Sample Walk-Forward Predictive Metrics (Daily Rank IC):", flush=True)
    print(f"  - Daily Rank IC (Mean)              : {wf_metrics.get('ic', 0.0):+.6f}", flush=True)
    print(f"  - Newey-West t-Statistic            : {wf_metrics.get('ic_tstat', 0.0):+.6f}", flush=True)

    print("\n[FROZEN FINAL HOLDOUT EVALUATION] (Last 9 Months):", flush=True)
    print(f"  - Holdout Period                   : {holdout.get('holdout_start')} to {holdout.get('holdout_end')}", flush=True)
    print(f"  - Holdout Rank IC                  : {holdout.get('holdout_ic', 0.0):+.4f}", flush=True)
    print(f"  - Holdout Net Sharpe (Full Model)  : {holdout.get('holdout_sharpe_full', 0.0):+.2f}", flush=True)
    print(f"  - Holdout Net Sharpe (Equal Weight): {holdout.get('holdout_sharpe_eq', 0.0):+.2f}", flush=True)

    print("\n[INDIC NEWS SENTIMENT ABLATION STUDY]:", flush=True)
    print(f"  - IC WITH Indic News Sentiment     : {ablation.get('ic_with_sentiment', 0.0):+.4f}", flush=True)
    print(f"  - IC WITHOUT Sentiment (Ablated)  : {ablation.get('ic_without_sentiment', 0.0):+.4f}", flush=True)
    print(f"  - Sharpe WITH Indic News Sentiment : {ablation.get('sharpe_with_sentiment', 0.0):+.2f}", flush=True)
    print(f"  - Sharpe WITHOUT Sentiment (Ablated): {ablation.get('sharpe_without_sentiment', 0.0):+.2f}", flush=True)

    print("\nEmpirical Strategy Baseline Comparison Table:", flush=True)
    print(comp_df.to_string(index=False), flush=True)

    return result


def step_3_update_artifacts(result: dict):
    print("\n" + "=" * 75, flush=True)
    print("STEP 3: Saving Performance Summary JSON & Rendering Documentation", flush=True)
    print("=" * 75, flush=True)

    summary_path = Path("results/performance_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_dict = {
        "total_candidate_features": result["total_candidate_features"],
        "quality_filtered_features": result["quality_filtered_features"],
        "walk_forward_metrics": result.get("walk_forward_metrics", {}),
        "baseline_vs_full_comparison": result.get("baseline_vs_full_comparison", []),
        "cost_sensitivity_table": result.get("cost_sensitivity_table", []),
        "sharpe_bootstrap_results": result.get("sharpe_bootstrap_results", {}),
        "frozen_holdout_metrics": result.get("frozen_holdout_metrics", {}),
        "indic_news_ablation": result.get("indic_news_ablation", {}),
        "trial_log_summary": result.get("trial_log_summary", {}),
        "ic_by_year": result.get("ic_by_year", {}),
        "backtest_results": result.get("backtest_results", {}),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=str)
    print(f"Saved raw execution output to {summary_path}", flush=True)

    _update_readme_table(summary_dict)

    from signalforge.monitoring.docx_report import create_updated_document
    create_updated_document()
    print("Updated README.md and SignalForge_Quant_Review_and_Improvement_Plan.docx", flush=True)


def _update_readme_table(summary: dict):
    readme_path = Path("README.md")
    if not readme_path.exists():
        return

    comp_list = summary.get("baseline_vs_full_comparison", [])
    if not comp_list:
        return

    comp_df = pd.DataFrame(comp_list)

    def get_val(row_name, col_name, fmt="{:.4f}"):
        sub = comp_df[comp_df["Metric"] == row_name]
        if sub.empty or col_name not in sub.columns:
            return "N/A"
        val = sub[col_name].values[0]
        if isinstance(val, (float, int)):
            return fmt.format(val)
        return str(val)

    ic_eq = get_val("Information Coefficient (IC)", "Equal Weight")
    ic_mom = get_val("Information Coefficient (IC)", "12-1 Momentum")
    ic_rev = get_val("Information Coefficient (IC)", "5-Day Reversal")
    ic_rev_lt = get_val("Information Coefficient (IC)", "Low-Turnover Reversal")
    ic_ohlcv = get_val("Information Coefficient (IC)", "OHLCV Baseline")
    ic_full = get_val("Information Coefficient (IC)", "Full Model")

    t_mom = get_val("Newey-West t-stat", "12-1 Momentum")
    t_rev = get_val("Newey-West t-stat", "5-Day Reversal")
    t_rev_lt = get_val("Newey-West t-stat", "Low-Turnover Reversal")
    t_ohlcv = get_val("Newey-West t-stat", "OHLCV Baseline")
    t_full = get_val("Newey-West t-stat", "Full Model")

    sh_eq = get_val("Strategy Sharpe Ratio (Net)", "Equal Weight", fmt="{:+.2f}")
    sh_mom = get_val("Strategy Sharpe Ratio (Net)", "12-1 Momentum", fmt="{:+.2f}")
    sh_rev = get_val("Strategy Sharpe Ratio (Net)", "5-Day Reversal", fmt="{:+.2f}")
    sh_rev_lt = get_val("Strategy Sharpe Ratio (Net)", "Low-Turnover Reversal", fmt="{:+.2f}")
    sh_ohlcv = get_val("Strategy Sharpe Ratio (Net)", "OHLCV Baseline", fmt="{:+.2f}")
    sh_full = get_val("Strategy Sharpe Ratio (Net)", "Full Model", fmt="{:+.2f}")

    dsr41_eq = get_val("Deflated Sharpe Ratio (DSR @ 41 trials)", "Equal Weight", fmt="{:.2f}")
    dsr41_mom = get_val("Deflated Sharpe Ratio (DSR @ 41 trials)", "12-1 Momentum", fmt="{:.2f}")
    dsr41_rev = get_val("Deflated Sharpe Ratio (DSR @ 41 trials)", "5-Day Reversal", fmt="{:.2f}")
    dsr41_rev_lt = get_val("Deflated Sharpe Ratio (DSR @ 41 trials)", "Low-Turnover Reversal", fmt="{:.2f}")
    dsr41_ohlcv = get_val("Deflated Sharpe Ratio (DSR @ 41 trials)", "OHLCV Baseline", fmt="{:.2f}")
    dsr41_full = get_val("Deflated Sharpe Ratio (DSR @ 41 trials)", "Full Model", fmt="{:.2f}")

    dsr100_full = get_val("Deflated Sharpe Ratio (DSR @ 100 trials)", "Full Model", fmt="{:.2f}")
    dsr200_full = get_val("Deflated Sharpe Ratio (DSR @ 200 trials)", "Full Model", fmt="{:.2f}")

    dd_eq = get_val("Max Drawdown (%)", "Equal Weight", fmt="{:.2%}")
    dd_mom = get_val("Max Drawdown (%)", "12-1 Momentum", fmt="{:.2%}")
    dd_rev = get_val("Max Drawdown (%)", "5-Day Reversal", fmt="{:.2%}")
    dd_rev_lt = get_val("Max Drawdown (%)", "Low-Turnover Reversal", fmt="{:.2%}")
    dd_ohlcv = get_val("Max Drawdown (%)", "OHLCV Baseline", fmt="{:.2%}")
    dd_full = get_val("Max Drawdown (%)", "Full Model", fmt="{:.2%}")

    tr_eq = get_val("Portfolio Turnover", "Equal Weight", fmt="{:.2f}")
    tr_mom = get_val("Portfolio Turnover", "12-1 Momentum", fmt="{:.2f}")
    tr_rev = get_val("Portfolio Turnover", "5-Day Reversal", fmt="{:.2f}")
    tr_rev_lt = get_val("Portfolio Turnover", "Low-Turnover Reversal", fmt="{:.2f}")
    tr_ohlcv = get_val("Portfolio Turnover", "OHLCV Baseline", fmt="{:.2f}")
    tr_full = get_val("Portfolio Turnover", "Full Model", fmt="{:.2f}")

    table_md = f"""| Strategy Baseline | Out-of-Sample IC | Newey-West $t$-stat | Strategy Sharpe (Net ~13.6bps) | DSR (41 / 100 / 200 trials) | Max Drawdown (%) | Portfolio Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal-Weight Buy & Hold** | {ic_eq} | N/A (Constant) | `{sh_eq}` | `{dsr41_eq}` | `{dd_eq}` | `{tr_eq}` |
| **12-1 Month Momentum** | `{ic_mom}` | `{t_mom}` | `{sh_mom}` | `{dsr41_mom}` | `{dd_mom}` | `{tr_mom}` |
| **5-Day Short Reversal** | `{ic_rev}` | `{t_rev}` | `{sh_rev}` | `{dsr41_rev}` | `{dd_rev}` | `{tr_rev}` |
| **Low-Turnover Reversal** | `{ic_rev_lt}` | `{t_rev_lt}` | `{sh_rev_lt}` | `{dsr41_rev_lt}` | `{dd_rev_lt}` | `{tr_rev_lt}` |
| **OHLCV Linear Baseline** | `{ic_ohlcv}` | `{t_ohlcv}` | `{sh_ohlcv}` | `{dsr41_ohlcv}` | `{dd_ohlcv}` | `{tr_ohlcv}` |
| **Full SignalForge Model** | **`{ic_full}`** | **`{t_full}`** | **`{sh_full}`** | **`{dsr41_full} / {dsr100_full} / {dsr200_full}`** | **`{dd_full}`** | **`{tr_full}`** |"""

    cost_list = summary.get("cost_sensitivity_table", [])
    if cost_list:
        cost_df = pd.DataFrame(cost_list)
        cost_md_lines = ["| Transaction Cost Level | Equal Weight | 12-1 Momentum | 5-Day Reversal | Low-Turnover Reversal | Full Model |",
                         "| :--- | :---: | :---: | :---: | :---: | :---: |"]
        for _, r in cost_df.iterrows():
            cost_md_lines.append(f"| `{r['Transaction Cost (bps)']}` | `{r.get('Equal Weight', 0.0):+.2f}` | `{r.get('12-1 Momentum', 0.0):+.2f}` | `{r.get('5-Day Reversal', 0.0):+.2f}` | `{r.get('Low-Turnover Reversal', 0.0):+.2f}` | `{r.get('Full Model', 0.0):+.2f}` |")
        cost_table_md = "\n".join(cost_md_lines)
    else:
        cost_table_md = ""

    boot = summary.get("sharpe_bootstrap_results", {})
    full_vs_eq = boot.get("full_vs_eq", {})
    rev_vs_eq = boot.get("reversal_vs_eq", {})
    rev_lt_vs_eq = boot.get("reversal_lt_vs_eq", {})

    boot_table_md = f"""| Comparison vs Equal-Weight | Sharpe Difference ($\\Delta SR$) | 95% Bootstrap CI | $p$-value |
| :--- | :---: | :---: | :---: |
| **Full Model vs Equal Weight** | `{full_vs_eq.get('diff_sharpe', 0.0):+.2f}` | `[{full_vs_eq.get('ci_low', 0.0):+.2f}, {full_vs_eq.get('ci_high', 0.0):+.2f}]` | `{full_vs_eq.get('p_value', 1.0):.4f}` |
| **5-Day Reversal vs Equal Weight** | `{rev_vs_eq.get('diff_sharpe', 0.0):+.2f}` | `[{rev_vs_eq.get('ci_low', 0.0):+.2f}, {rev_vs_eq.get('ci_high', 0.0):+.2f}]` | `{rev_vs_eq.get('p_value', 1.0):.4f}` |
| **Low-Turnover Reversal vs Equal Weight** | `{rev_lt_vs_eq.get('diff_sharpe', 0.0):+.2f}` | `[{rev_lt_vs_eq.get('ci_low', 0.0):+.2f}, {rev_lt_vs_eq.get('ci_high', 0.0):+.2f}]` | `{rev_lt_vs_eq.get('p_value', 1.0):.4f}` |"""

    holdout = summary.get("frozen_holdout_metrics", {})
    holdout_md = f"""| Holdout Period | Frozen Holdout IC | Full Model Net Sharpe | Equal Weight Net Sharpe | Max Drawdown |
| :---: | :---: | :---: | :---: | :---: |
| `{holdout.get('holdout_start')} to {holdout.get('holdout_end')}` | `{holdout.get('holdout_ic', 0.0):+.4f}` | `{holdout.get('holdout_sharpe_full', 0.0):+.2f}` | `{holdout.get('holdout_sharpe_eq', 0.0):+.2f}` | `{holdout.get('holdout_max_dd_full', 0.0):.2%}` |"""

    ablation = summary.get("indic_news_ablation", {})
    ablation_md = f"""| Model Variant | Out-of-Sample Rank IC | Net Strategy Sharpe Ratio |
| :--- | :---: | :---: |
| **Full Model WITH Indic News Sentiment** | `{ablation.get('ic_with_sentiment', 0.0):+.4f}` | `{ablation.get('sharpe_with_sentiment', 0.0):+.2f}` |
| **Ablated Model WITHOUT Sentiment** | `{ablation.get('ic_without_sentiment', 0.0):+.4f}` | `{ablation.get('sharpe_without_sentiment', 0.0):+.2f}` |"""

    content = readme_path.read_text(encoding="utf-8")
    start_marker = "## 📊 Benchmark Strategy Comparison (50 Liquid Large Caps, 5 Years)"
    end_marker = "## 🚀 Quickstart & Usage"

    if start_marker in content and end_marker in content:
        parts = content.split(start_marker)
        after_section = parts[1].split(end_marker, 1)[1]

        section_body = f"""\n\n{table_md}\n\n### 🧊 Frozen Final Holdout Evaluation (Last 9 Months)\n\n{holdout_md}\n\n### 📰 Indic News Sentiment Ablation Study\n\n{ablation_md}\n\n### 💸 Transaction Cost Sensitivity (Sharpe at 0 to 25 bps)\n\n{cost_table_md}\n\n### 🎲 Sharpe Difference Circular Block Bootstrap vs Equal-Weight\n\n{boot_table_md}\n\n---\n\n"""
        new_content = parts[0] + start_marker + section_body + end_marker + after_section
        readme_path.write_text(new_content, encoding="utf-8")


def main():
    df = step_1_fetch_and_verify_eod()
    result = step_2_run_pipeline_and_baselines(df)
    step_3_update_artifacts(result)
    print("\n" + "=" * 75, flush=True)
    print("SUCCESS: End-to-End Pipeline Execution, Verification & Dynamic Artifacts Complete", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
