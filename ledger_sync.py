from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from signalforge.live.paper import load_ledger, save_ledger


def export_ledger_to_csv(ledger: dict[str, Any] | str | Path, csv_path: str | Path = "desk/paper-ledger.csv") -> Path:
    """Exports paper/ledger.json to a desk-compatible paper-ledger.csv format."""
    if isinstance(ledger, (str, Path)):
        ledger_data = load_ledger(ledger)
    else:
        ledger_data = ledger

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cash = float(ledger_data.get("cash", 0.0))
    peak_equity = float(ledger_data.get("peak_equity", cash))
    last_equity = float(ledger_data.get("last_equity", cash))
    paused = bool(ledger_data.get("paused", False))
    pause_reason = str(ledger_data.get("pause_reason", ""))

    positions = ledger_data.get("positions", {})
    if not positions:
        # Single summary row for cash-only position
        rows.append({
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": "CASH",
            "quantity": 0,
            "avg_price": 1.0,
            "notional": cash,
            "equity": last_equity,
            "cash": cash,
            "peak_equity": peak_equity,
            "paused": paused,
            "pause_reason": pause_reason,
        })
    else:
        for sym, pos in positions.items():
            qty = float(pos.get("qty", 0.0))
            avg_px = float(pos.get("avg_price", 0.0))
            rows.append({
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym,
                "quantity": qty,
                "avg_price": avg_px,
                "notional": qty * avg_px,
                "equity": last_equity,
                "cash": cash,
                "peak_equity": peak_equity,
                "paused": paused,
                "pause_reason": pause_reason,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path


def import_csv_to_ledger(csv_path: str | Path, ledger_json_path: str | Path = "paper/ledger.json") -> dict[str, Any]:
    """Imports desk/paper-ledger.csv into paper/ledger.json format."""
    in_path = Path(csv_path)
    if not in_path.exists():
        raise FileNotFoundError(f"CSV ledger not found at {csv_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise ValueError("CSV ledger file is empty.")

    last_row = df.iloc[-1]
    cash = float(last_row.get("cash", 1_000_000.0))
    peak_equity = float(last_row.get("peak_equity", cash))
    paused = bool(last_row.get("paused", False))
    pause_reason = str(last_row.get("pause_reason", ""))

    positions: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        sym = str(row["symbol"]).strip().upper()
        if sym != "CASH" and float(row.get("quantity", 0)) > 0:
            positions[sym] = {
                "qty": float(row["quantity"]),
                "avg_price": float(row.get("avg_price", 0.0)),
            }

    ledger_data = {
        "starting_cash": float(df.iloc[0].get("cash", cash)),
        "cash": cash,
        "positions": positions,
        "pending": None,
        "peak_equity": peak_equity,
        "last_equity": float(last_row.get("equity", cash)),
        "paused": paused,
        "pause_reason": pause_reason,
        "cost_bps": 10.0,
        "history": [],
    }

    save_ledger(ledger_data, ledger_json_path)
    return ledger_data
