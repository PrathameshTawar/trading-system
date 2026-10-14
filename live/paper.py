from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signalforge.live.alerts import send_alert
from signalforge.live.validation import is_nse_trading_day, validate_daily_data


import subprocess
from signalforge.backtesting.costs import compute_indian_transaction_cost, get_one_way_cost_bps


def commit_and_push_signal_log(file_path: str | Path) -> bool:
    """Automatically git commit & push signal log before market open to ensure tamper-proof proof-of-signal."""
    p = Path(file_path)
    try:
        cmd_add = ["git", "add", str(p)]
        subprocess.run(cmd_add, capture_output=True, check=False)
        
        cmd_commit = ["git", "commit", "-m", f"Pre-open signal log chain commit [{p.name}]"]
        res_commit = subprocess.run(cmd_commit, capture_output=True, text=True, check=False)
        
        cmd_push = ["git", "push"]
        res_push = subprocess.run(cmd_push, capture_output=True, text=True, check=False)
        return res_commit.returncode == 0 or "nothing to commit" in res_commit.stdout.lower()
    except Exception:
        return False


def log_signals(signals: pd.DataFrame, run_date: str, path: str | Path = "paper/signal_log.jsonl") -> bool:
    """Cryptographic chained append-only signal commitment log with SHA-256 hash chaining.
    
    Idempotent: Never rewrites a past day's signal log. Automatically commits & pushes before open.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    prev_hash = "0" * 64

    if p.exists():
        lines = p.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                try:
                    obj = json.loads(line)
                    seen.add(obj["date"])
                    prev_hash = obj.get("sha256", prev_hash)
                except Exception:
                    pass

    if run_date in seen:
        return False  # Idempotent: never rewrite a past day

    if not signals.empty and "symbol" in signals.columns:
        cols = [c for c in ["symbol", "prediction", "weight"] if c in signals.columns]
        rows = signals[cols].round(8).to_dict("records")
    else:
        rows = []

    payload = f"{prev_hash}|{run_date}|" + json.dumps(rows, sort_keys=True)
    curr_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "date": run_date,
            "previous_hash": prev_hash,
            "sha256": curr_hash,
            "signals": rows
        }) + "\n")

    # Auto commit and push to version control before market open
    commit_and_push_signal_log(p)
    return True


def _empty_ledger(starting_cash: float, cost_bps: float) -> dict:
    return {
        "starting_cash": starting_cash,
        "cash": starting_cash,
        "positions": {},
        "pending": None,
        "peak_equity": starting_cash,
        "paused": False,
        "pause_reason": "",
        "cost_bps": cost_bps,
        "history": [],
    }


def load_ledger(path: str | Path, starting_cash: float = 1_000_000.0, cost_bps: float = 10.0) -> dict:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return _empty_ledger(starting_cash, cost_bps)
    with ledger_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_ledger(ledger: dict, path: str | Path) -> Path:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2, default=str)
    return ledger_path


def _session_bars(market_df: pd.DataFrame, session: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(market_df["timestamp"]).dt.normalize()
    return market_df.loc[ts == pd.to_datetime(session).normalize()].copy()


def _next_session(market_df: pd.DataFrame, after: pd.Timestamp) -> pd.Timestamp | None:
    dates = sorted(pd.to_datetime(market_df["timestamp"]).dt.normalize().unique())
    after_n = pd.to_datetime(after).normalize()
    later = [d for d in dates if d > after_n]
    return later[0] if later else None


def mark_to_market(ledger: dict, bars: pd.DataFrame) -> float:
    px = {row["symbol"]: float(row["close"]) for _, row in bars.iterrows()}
    equity = float(ledger["cash"])
    for symbol, pos in ledger.get("positions", {}).items():
        qty = float(pos.get("qty", 0))
        if qty and symbol in px:
            equity += qty * px[symbol]
    return equity


def _rebalance_at_open(
    ledger: dict,
    bars: pd.DataFrame,
    targets: dict[str, float],
    adv_20d: dict[str, float] | None = None,
) -> dict:
    adv = adv_20d or {}
    opens = {row["symbol"]: float(row["open"]) for _, row in bars.iterrows()}
    equity = mark_to_market(ledger, bars)
    cash = float(ledger["cash"])
    new_positions: dict[str, dict] = {}
    traded_notional = 0.0
    total_cost = 0.0

    universe = set(opens) | set(ledger.get("positions", {})) | set(targets)
    for symbol in sorted(universe):
        open_px = opens.get(symbol)
        if open_px is None or open_px <= 0:
            if symbol in ledger.get("positions", {}):
                new_positions[symbol] = ledger["positions"][symbol]
            continue

        curr_qty = float(ledger.get("positions", {}).get(symbol, {}).get("qty", 0.0))
        target_weight = float(targets.get(symbol, 0.0))
        target_notional = equity * target_weight

        # Minimum order value check (₹5,000)
        if target_notional > 0 and target_notional < 5000.0:
            target_qty = curr_qty
        else:
            target_qty = float(int(target_notional / open_px)) if target_weight > 0 else 0.0

        delta = target_qty - curr_qty
        if abs(delta) < 1:
            if curr_qty > 0:
                new_positions[symbol] = ledger["positions"][symbol]
            continue

        # Calculate order notional & volume-dependent slippage
        notional = abs(delta) * open_px
        adv_notional = float(adv.get(symbol, 10_000_000.0))
        participation = min(1.0, notional / max(adv_notional, 100_000.0))
        slippage_bps = 5.0 + 10.0 * np.sqrt(participation)
        fill_px = open_px * (1.0 + (slippage_bps / 10000.0) if delta > 0 else 1.0 - (slippage_bps / 10000.0))

        action = "buy" if delta > 0 else "sell"
        cost = compute_indian_transaction_cost(notional, action=action)

        cash -= delta * fill_px + cost
        traded_notional += notional
        total_cost += cost

        if target_qty > 0:
            new_positions[symbol] = {"qty": target_qty, "avg_price": fill_px}

    ledger["cash"] = max(0.0, cash)
    ledger["positions"] = new_positions
    ledger["last_turnover_notional"] = traded_notional
    ledger["last_cost_incurred"] = total_cost
    return ledger


def paper_step(
    market_df: pd.DataFrame,
    ledger: dict,
    signals: pd.DataFrame | None,
    *,
    max_drawdown_pause: float = 0.10,
    flatten_on_pause: bool = True,
    signal_log_path: str | Path = "paper/signal_log.jsonl",
    check_holidays: bool = False,
) -> dict:
    """Execute paper trading step.
    
    1. Validates daily market data
    2. Checks NSE trading holiday calendar (if check_holidays=True)
    3. Fills yesterday's targets at today's open net of realistic Indian costs + volume slippage
    4. Marks portfolio to market at close
    5. Keyed idempotently by session date (prevents duplicate daily entries)
    6. Logs signals to append-only cryptographic JSONL log
    """
    market_df = market_df.copy()
    market_df["timestamp"] = pd.to_datetime(market_df["timestamp"])
    asof = pd.to_datetime(market_df["timestamp"]).dt.normalize().max()
    date_str = str(asof.date())

    # Check NSE Trading Holiday
    if check_holidays and not is_nse_trading_day(asof):
        send_alert("INFO", f"Session {date_str} is an NSE holiday / weekend; skipping paper step.")
        return ledger

    # Validate daily data
    valid, rejections = validate_daily_data(market_df)
    if not valid:
        send_alert("WARNING", f"Data validation failed on {date_str}; skipping trade execution.", {"reasons": rejections})
        return ledger

    today_bars = _session_bars(market_df, asof)
    if today_bars.empty:
        raise ValueError(f"No bars for session {date_str}; cannot paper trade.")

    # Compute 20-day ADV notional lookup
    adv_20d = {}
    if "volume" in market_df.columns and "close" in market_df.columns:
        market_df["notional"] = market_df["close"] * market_df["volume"]
        adv_series = market_df.groupby("symbol")["notional"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        market_df["adv_20d"] = adv_series
        adv_20d = dict(today_bars.merge(market_df[["symbol", "timestamp", "adv_20d"]], on=["symbol", "timestamp"], how="left")[["symbol", "adv_20d"]].values)

    pending = ledger.get("pending")
    if pending:
        fill_session = _next_session(market_df, pending["signal_date"])
        if fill_session is not None and fill_session == asof:
            targets = {row["symbol"]: float(row["weight"]) for row in pending.get("targets", [])}
            if ledger.get("paused") and flatten_on_pause:
                targets = {}
            fill_bars = _session_bars(market_df, fill_session)
            ledger = _rebalance_at_open(ledger, fill_bars, targets, adv_20d=adv_20d)
            ledger["last_fill_session"] = date_str
            ledger["pending"] = None

    equity = mark_to_market(ledger, today_bars)
    peak = max(float(ledger.get("peak_equity", equity)), equity)
    ledger["peak_equity"] = peak
    drawdown = (equity / peak - 1.0) if peak else 0.0

    if drawdown <= -abs(max_drawdown_pause):
        if not ledger.get("paused"):
            send_alert("CRITICAL", f"Kill-switch triggered! Max drawdown {drawdown:.2%} breached peak equity.", {"equity": equity, "peak": peak})
        ledger["paused"] = True
        ledger["pause_reason"] = f"max drawdown {drawdown:.2%} breached"

    # Idempotent ledger history update (keyed by date_str)
    hist_entry = {
        "date": date_str,
        "equity": equity,
        "cash": ledger["cash"],
        "drawdown": drawdown,
        "paused": ledger["paused"],
        "positions": {k: v.get("qty") for k, v in ledger.get("positions", {}).items()},
    }
    existing_hist = ledger.get("history", [])
    updated_hist = [h for h in existing_hist if h.get("date") != date_str]
    updated_hist.append(hist_entry)
    ledger["history"] = updated_hist

    if signals is not None and not signals.empty and not ledger.get("paused"):
        ledger["pending"] = {
            "signal_date": date_str,
            "targets": [
                {"symbol": row["symbol"], "weight": float(row["weight"]), "prediction": float(row["prediction"])}
                for _, row in signals.iterrows()
            ],
        }
        # Commit signals to cryptographic append-only signal log
        log_signals(signals, date_str, path=signal_log_path)
    elif ledger.get("paused"):
        ledger["pending"] = {
            "signal_date": date_str,
            "targets": [],
        }

    ledger["last_equity"] = equity
    ledger["last_session"] = date_str
    return ledger


def run_shadow_portfolios(
    market_df: pd.DataFrame,
    bundle: Any,
    base_dir: str | Path = "paper/shadow",
) -> dict[str, dict]:
    """Run side-by-side shadow portfolios (Full Model, Equal Weight, 5-Day Reversal, 12-1 Momentum)."""
    b_dir = Path(base_dir)
    b_dir.mkdir(parents=True, exist_ok=True)

    from signalforge.live.positions import scores_to_cash_weights
    from signalforge.live.score import score_universe

    model_signals, _, _ = score_universe(market_df, bundle)
    latest_bars = market_df[market_df["timestamp"] == market_df["timestamp"].max()].copy()

    # 1. Equal Weight Signal
    eq_signals = latest_bars.copy()
    eq_signals["prediction"] = 1.0
    eq_signals = scores_to_cash_weights(eq_signals, max_names=len(eq_signals), max_weight=1.0 / max(len(eq_signals), 1))

    # 2. 5-Day Reversal Signal (-return_5d)
    rev_signals = latest_bars.copy()
    if "return_5d" in rev_signals.columns:
        rev_signals["prediction"] = -rev_signals["return_5d"]
    else:
        rev_signals["prediction"] = -rev_signals.get("return_1d", 0.0)
    rev_signals = scores_to_cash_weights(rev_signals, max_names=10, max_weight=0.05)

    # 3. 12-1 Momentum Signal
    mom_signals = latest_bars.copy()
    mom_signals["prediction"] = mom_signals.get("momentum_12_1", mom_signals.get("return_60d", 0.0))
    mom_signals = scores_to_cash_weights(mom_signals, max_names=10, max_weight=0.05)

    shadow_configs = {
        "full_model": (model_signals, b_dir / "ledger_full_model.json"),
        "equal_weight": (eq_signals, b_dir / "ledger_equal_weight.json"),
        "reversal_5d": (rev_signals, b_dir / "ledger_reversal_5d.json"),
        "momentum_12_1": (mom_signals, b_dir / "ledger_momentum_12_1.json"),
    }

    results = {}
    for strat_name, (sig_df, path) in shadow_configs.items():
        ledger = load_ledger(path, starting_cash=100_000.0, cost_bps=10.0)
        updated_ledger = paper_step(market_df, ledger, sig_df, signal_log_path=b_dir / f"signal_log_{strat_name}.jsonl")
        save_ledger(updated_ledger, path)
        results[strat_name] = updated_ledger

    return results
