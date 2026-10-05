from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from signalforge.live.alerts import send_alert
from signalforge.live.paper import commit_and_push_signal_log, log_signals, paper_step
from signalforge.live.score import score_universe


def generate_daily_live_report(
    market_df: pd.DataFrame,
    ledger: dict,
    output_dir: str | Path = "results",
) -> dict:
    """Generate daily live monitoring report tracking:
    - Strategy NAV vs Equal-Weight Benchmark NAV vs ^NSEI Index NAV
    - Live rank IC against realized 5-day returns
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if market_df.empty or "timestamp" not in market_df.columns or "close" not in market_df.columns:
        return {}

    ts_df = market_df.copy()
    ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
    latest_date = ts_df["timestamp"].max()
    date_str = str(latest_date.date())

    # 1. NAV Tracking: Strategy NAV vs Equal-Weight NAV vs ^NSEI NAV
    hist = ledger.get("history", [])
    if hist:
        strat_nav = hist[-1].get("equity", ledger.get("starting_cash", 1_000_000.0))
        start_equity = ledger.get("starting_cash", 1_000_000.0)
        strat_ret = (strat_nav / start_equity - 1.0) if start_equity else 0.0
    else:
        strat_nav = ledger.get("cash", 1_000_000.0)
        strat_ret = 0.0

    # Benchmark ^NSEI series
    if "nifty_close" in ts_df.columns:
        nifty_series = ts_df.groupby("timestamp")["nifty_close"].mean().sort_index()
        nifty_start = nifty_series.iloc[0] if len(nifty_series) > 0 else 1.0
        nifty_end = nifty_series.iloc[-1] if len(nifty_series) > 0 else 1.0
        nifty_ret = (nifty_end / nifty_start - 1.0) if nifty_start else 0.0
    else:
        nifty_ret = 0.0

    # Equal Weight benchmark return
    eq_close = ts_df.groupby("timestamp")["close"].mean().sort_index()
    eq_start = eq_close.iloc[0] if len(eq_close) > 0 else 1.0
    eq_end = eq_close.iloc[-1] if len(eq_close) > 0 else 1.0
    eq_ret = (eq_end / eq_start - 1.0) if eq_start else 0.0

    # 2. Live Rank IC against realized 5-day returns
    live_rank_ic = 0.0
    if "prediction" in ts_df.columns and "future_return_5d" in ts_df.columns:
        recent_df = ts_df.dropna(subset=["prediction", "future_return_5d"]).copy()
        if not recent_df.empty:
            def _group_spearman(sub):
                if len(sub) < 3 or sub["prediction"].nunique() <= 1 or sub["future_return_5d"].nunique() <= 1:
                    return np.nan
                corr, _ = spearmanr(sub["prediction"], sub["future_return_5d"])
                return float(corr) if not np.isnan(corr) else np.nan

            daily_ics = recent_df.groupby("timestamp").apply(_group_spearman).dropna()
            live_rank_ic = float(daily_ics.mean()) if not daily_ics.empty else 0.0

    report_data = {
        "report_date": date_str,
        "strategy_nav": float(strat_nav),
        "strategy_cumulative_return": float(strat_ret),
        "equal_weight_return": float(eq_ret),
        "nifty_index_return": float(nifty_ret),
        "live_rank_ic_5d": float(live_rank_ic),
        "portfolio_drawdown": float(hist[-1].get("drawdown", 0.0)) if hist else 0.0,
        "is_paused": bool(ledger.get("paused", False)),
    }

    # Save JSON report
    json_path = out_p / "live_daily_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown report
    md_path = out_p / "live_daily_report.md"
    md_content = f"""# 📈 SignalForge Daily Live Performance Report ({date_str})

| Metric | Value |
| :--- | :---: |
| **Strategy NAV** | ₹{strat_nav:,.2f} (`{strat_ret:+.2%}`) |
| **Equal-Weight Benchmark Return** | `{eq_ret:+.2%}` |
| **^NSEI Index Return** | `{nifty_ret:+.2%}` |
| **Live Rank IC (vs Realized 5D Returns)** | `{live_rank_ic:+.4f}` |
| **Current Drawdown** | `{report_data['portfolio_drawdown']:.2%}` |
| **Execution Status** | `{"PAUSED" if report_data['is_paused'] else "ACTIVE"}` |
"""
    md_path.write_text(md_content, encoding="utf-8")
    return report_data


class DailyLiveScheduler:
    """Production Daily Live Trading & Monitoring Scheduler."""

    def __init__(self, ledger_path: str | Path = "paper/ledger.json", signal_log_path: str | Path = "paper/signal_log.jsonl"):
        self.ledger_path = Path(ledger_path)
        self.signal_log_path = Path(signal_log_path)

    def run_pre_market_job(self, market_df: pd.DataFrame, bundle: any) -> dict:
        """Pre-market job (08:45 AM IST): generate predictions, update signal hash chain, commit & push to git."""
        asof_date = str(pd.to_datetime(market_df["timestamp"]).max().date())
        send_alert("INFO", f"Running Pre-Market Job for session {asof_date}")

        # Score universe
        model_signals, _, _ = score_universe(market_df, bundle)

        # Append to cryptographic chained signal log & auto commit/push to git
        logged = log_signals(model_signals, asof_date, path=self.signal_log_path)
        if logged:
            commit_and_push_signal_log(self.signal_log_path)
            send_alert("INFO", f"Pre-market signal log committed & pushed to git for {asof_date}")

        return {"date": asof_date, "signals_count": len(model_signals)}

    def run_post_market_job(self, market_df: pd.DataFrame, bundle: any) -> dict:
        """Post-market job (15:45 PM IST): execute paper step, update ledger, generate daily NAV report."""
        from signalforge.live.paper import load_ledger, paper_step, save_ledger

        asof_date = str(pd.to_datetime(market_df["timestamp"]).max().date())
        send_alert("INFO", f"Running Post-Market Job for session {asof_date}")

        ledger = load_ledger(self.ledger_path)
        model_signals, _, _ = score_universe(market_df, bundle)

        updated_ledger = paper_step(market_df, ledger, model_signals, signal_log_path=self.signal_log_path)
        save_ledger(updated_ledger, self.ledger_path)

        report = generate_daily_live_report(market_df, updated_ledger)
        return report
